"""Derive reviewer-facing timing/call/token/storage tables from raw artifacts."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import statistics

import numpy as np

from .run import read_json, read_jsonl
from .transport import file_hash, write_json


def distribution(values):
    return {
        "count": len(values), "total_s": sum(values),
        "mean_s": statistics.mean(values) if values else None,
        "p50_s": float(np.percentile(values, 50)) if values else None,
        "p95_s": float(np.percentile(values, 95)) if values else None,
        "min_s": min(values) if values else None,
        "max_s": max(values) if values else None,
    }


def query_wall_attempts(phases, campaign_attempts, dataset):
    """Recover every query phase, including work before an interrupted resume."""
    events = sorted(
        (row for row in phases if row.get("dataset") == dataset
         and row.get("phase") in ("query_phase_start", "query_phase_complete")),
        key=lambda row: row["time_unix"])
    result = []
    for index, event in enumerate(events):
        if event["phase"] != "query_phase_start":
            continue
        started = event["time_unix"]
        following = events[index + 1] if index + 1 < len(events) else None
        if following and following["phase"] == "query_phase_complete":
            result.append({
                "started_unix": started, "finished_unix": following["time_unix"],
                "status": "completed", "elapsed_s": following["elapsed_s"],
                "measurement": "recorded_monotonic_query_phase_elapsed"})
            continue
        terminals = [
            row for row in campaign_attempts
            if row["started_unix"] <= started <= row["finished_unix"]
            and (following is None or row["finished_unix"] <= following["time_unix"])]
        if terminals:
            terminal = min(terminals, key=lambda row: row["finished_unix"])
            result.append({
                "started_unix": started, "finished_unix": terminal["finished_unix"],
                "status": terminal["status"],
                "elapsed_s": terminal["finished_unix"] - started,
                "measurement": "phase_start_to_failed_invocation_exit_wall_timestamps"})
        else:
            result.append({
                "started_unix": started, "status": "unresolved", "elapsed_s": None})
    return result


def ledger(calls, hits):
    totals = Counter(attempts=0, successful_calls=0, failed_attempts=0,
                     input_tokens=0, output_tokens=0, total_tokens=0,
                     cached_input_tokens=0, attempts_without_reported_usage=0,
                     provider_cost_records=0, provider_filtered_responses=0,
                     truncated_output_responses=0, incomplete_output_responses=0,
                     structured_provider_error_responses=0,
                     text_retry_attempts=0, text_retry_backoff_s=0,
                     preserved_text_failure_artifacts=0)
    reported_cost = 0.0
    for call in calls:
        totals["attempts"] += 1
        totals["successful_calls" if call["status"] == "completed" else "failed_attempts"] += 1
        totals["provider_filtered_responses"] += (
            call["status"] == "provider_filtered"
            or call.get("error") in ("completion_content_filter", "completion_refusal"))
        totals["truncated_output_responses"] += (
            call["status"] == "output_truncated" or call.get("error") == "completion_length")
        totals["incomplete_output_responses"] += (
            call["status"] == "output_incomplete"
            or call.get("error") in ("completion_None", "completion_missing_finish_reason"))
        totals["structured_provider_error_responses"] += (
            call["status"] == "failed" and call.get("finish_reason") == "error")
        totals["text_retry_attempts"] += call.get("attempt", 1) > 1
        totals["text_retry_backoff_s"] += call.get("retry_delay_s", 0)
        totals["preserved_text_failure_artifacts"] += bool(call.get("failure_artifact"))
        usage = call.get("usage") or {}
        if not usage:
            totals["attempts_without_reported_usage"] += 1
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            totals[key] += usage.get(key, 0) or 0
        totals["cached_input_tokens"] += (
            usage.get("input_tokens_details") or {}).get("cached_tokens", 0) or 0
        cost = (call.get("usage_raw") or {}).get("cost")
        if isinstance(cost, (int, float)):
            reported_cost += cost
            totals["provider_cost_records"] += 1
    totals["local_response_cache_hits"] = len(hits)
    totals["local_filtered_response_cache_hits"] = sum(
        hit.get("status") == "provider_filtered" for hit in hits)
    totals["local_truncated_response_cache_hits"] = sum(
        hit.get("status") == "output_truncated" for hit in hits)
    totals["local_incomplete_response_cache_hits"] = sum(
        hit.get("status") == "output_incomplete" for hit in hits)
    return {
        **dict(totals),
        "provider_reported_cost_sum": reported_cost,
        "provider_cost_note": "Sum of OpenRouter usage.cost; no inferred price or timeout charge.",
        "successful_http_latency": distribution([
            call["elapsed_s"] for call in calls if call["status"] == "completed"]),
    }


def build_efficiency(campaign, verified_rows):
    def optional(name):
        return read_jsonl(campaign / name) if (campaign / name).exists() else []
    calls, hits = optional("api_calls.jsonl"), optional("cache_hits.jsonl")
    embed = optional("embedding_invocations.jsonl")
    embed_http = optional("embedding_http_attempts.jsonl")
    resources, phases = optional("resource_samples.jsonl"), optional("phase_events.jsonl")
    campaign_attempts = optional("campaign_attempts.jsonl")
    rows = []
    for verified in verified_rows:
        dataset = verified["dataset"]
        root = campaign / dataset
        predictions = [read_json(path) for path in (root / "predictions").glob("*.json")]
        relevant = [record for record in calls if record.get("dataset") == dataset]
        relevant_hits = [record for record in hits if record.get("dataset") == dataset]
        relevant_keys = {call["request_sha256"] for call in relevant}
        by_stage = {}
        for stage in sorted({record["stage"] for record in relevant + relevant_hits}):
            by_stage[stage] = ledger(
                [record for record in relevant if record["stage"] == stage],
                [record for record in relevant_hits if record["stage"] == stage])
        by_phase = {
            phase: ledger(
                [record for record in relevant if bool(record.get("item_id")) == query],
                [record for record in relevant_hits if bool(record.get("item_id")) == query])
            for phase, query in [("construction", False), ("query", True)]
        }
        timing = (read_json(root / "timing.json")
                  if (root / "timing.json").exists() else {})
        query_attempts = query_wall_attempts(phases, campaign_attempts, dataset)
        if any(row["elapsed_s"] is None for row in query_attempts):
            raise RuntimeError("Unresolved query phase timing: " + dataset)
        query_all_attempts_s = sum(row["elapsed_s"] for row in query_attempts)
        samples = [sample for sample in resources if sample.get("dataset") == dataset]
        embeddings = [record for record in embed if record.get("dataset") == dataset]
        embedding_http = [record for record in embed_http
                          if record.get("dataset") == dataset]
        row = {
            "dataset": dataset, "questions": verified["completed"],
            "primary_em": verified["primary_em"], "primary_f1": verified["primary_f1"],
            "wall_time": timing,
            "construction_all_attempts_s": verified["total_construction_attempt_elapsed_s"],
            "query_phase_attempts": query_attempts,
            "query_all_attempts_wall_s": query_all_attempts_s,
            "query_all_attempts_questions_per_s": (
                len(predictions) / query_all_attempts_s if query_all_attempts_s else None),
            "query_latency": {
                key: distribution([p[field] for p in predictions if field in p])
                for key, field in [
                    ("end_to_end", "latency_s"),
                    ("retrieval_including_query_extraction", "retrieval_latency_s"),
                    ("answer_generation", "answer_generation_latency_s"),
                ]
            },
            "api": ledger(relevant, relevant_hits),
            "api_by_pipeline_phase": by_phase, "api_by_call_stage": by_stage,
            "embedding": {
                "logical_invocations": len(embeddings),
                "logical_texts": sum(record["texts"] for record in embeddings),
                "new_unique_texts": sum(record["new_unique_texts"] for record in embeddings),
                "cached_or_repeated_texts": sum(
                    record["cached_or_repeated_texts"] for record in embeddings),
                "remote_loopback_batch_requests": verified["embedding_requests"],
                "http_attempts_since_instrumentation": len(embedding_http),
                "http_successes_since_instrumentation": sum(
                    record["status"] == "completed" for record in embedding_http),
                "http_failures_since_instrumentation": sum(
                    record["status"] != "completed" for record in embedding_http),
                "http_retries_since_instrumentation": sum(
                    record["attempt"] > 1 for record in embedding_http),
                "http_retry_backoff_s": sum(
                    record["retry_delay_s"] for record in embedding_http),
                "http_failed_attempt_wall_s": sum(
                    record["elapsed_s"] for record in embedding_http
                    if record["status"] != "completed"),
                "http_status_counts_since_instrumentation": dict(Counter(
                    str(record.get("http_status", record.get("error")))
                    for record in embedding_http)),
                "overflow_fallback_completed_batches": sum(
                    record.get("truncate") is True
                    and record.get("status") == "completed"
                    and bool(record.get("overflow_artifact"))
                    for record in embedding_http),
                "overflow_fallback_input_texts": sum(
                    record["texts"] for record in embedding_http
                    if record.get("truncate") is True
                    and record.get("status") == "completed"
                    and record.get("overflow_artifact")),
                "overflow_fallback_single_input_batches": sum(
                    record.get("truncate") is True
                    and record.get("status") == "completed"
                    and bool(record.get("overflow_artifact"))
                    and record["texts"] == 1 for record in embedding_http),
                "overflow_policy": read_json(campaign / "provenance.json")[
                    "config"].get("embedding_overflow_policy"),
                "http_observation_scope": (
                    "Exact-attempt logging starts with the embedding recovery. "
                    "Older successful batches remain in embedding_calls.jsonl; "
                    "the original HTTP 400 body/failed-request count was not captured. "
                    "Earlier failed construction wall time is included."),
                "invocation_wall_time": distribution([record["elapsed_s"] for record in embeddings]),
                "scope_note": "BGE-M3 via separate Ollama service; dense vectors only.",
            },
            "storage": {
                "persistent_index_bytes": verified["index_storage_bytes"],
                "index_files_bytes": verified["index_files_bytes"],
                "response_cache_bytes": sum(
                    path.stat().st_size for path in (campaign / "cache/responses").glob("*.json")
                    if path.stem in relevant_keys),
                "response_cache_note": "Reproduction cache; excluded from deployed index size.",
            },
            "graph": verified["graph"],
            "provider_refusals": verified.get("provider_refusals", {}),
            "output_truncations": verified.get("output_truncations", {}),
            "incomplete_outputs": verified.get("incomplete_outputs", {}),
            "invalid_answer_schema_questions": verified.get("invalid_answer_schema_questions", 0),
            "path_question_repository": {
                "applicable": False, "count": 0,
                "reason": "Official HyperGraphRAG does not construct a path-question repository.",
            },
            "resource_observations": {
                "samples": len(samples),
                "peak_worker_resident_bytes": max(
                    (sample.get("peak_resident_bytes", 0) for sample in samples), default=0),
                "peak_worker_private_bytes": max(
                    (sample.get("private_bytes", 0) for sample in samples), default=0),
                "peak_llm_in_flight": max(
                    (sample.get("llm_peak_active", 0) for sample in samples), default=0),
                "gpu_scope": "Shared GPU; raw utilization and memory in resource_samples.jsonl.",
            },
            "phase_events": [event for event in phases if event.get("dataset") == dataset],
            "concurrency": verified["concurrent_query_workers"],
        }
        rows.append(row)
    record = {
        "datasets": rows,
        "latency_definition": (
            "Per-question monotonic wall time includes query extraction, local embedding, "
            "graph/vector retrieval, context assembly and answer API time, including bounded "
            "retries. Initial admission wait before a query worker is available is excluded. "
            "Also report query-phase wall time and throughput separately."),
        "measurement_limits": [
            "Single run at documented concurrency on a shared GPU and remote API.",
            "Cache hits and all observed retries are reported separately.",
            "Unreported server-side usage for failed/time-out attempts remains unknown.",
            "Only pooled candidate-passage corpora; no full-Wikipedia scalability claim.",
            "Worker RSS excludes the external Ollama process; GPU observations are device-wide.",
            "Provider-filtered outputs and their cached replays are reported separately; no questions are removed.",
            "Missing/null finish reasons are separately reported as incomplete outputs of unknown cause, with no partial-text use or retry.",
            "Embedding HTTP attempts before the transport recovery were not fully instrumented; earlier failed build wall time is preserved.",
            "query_all_attempts_wall_s includes interrupted query phases and resumed work; wall_time.query_phase_wall_s describes the last invocation only. Recovery downtime is excluded.",
            "Exact context-overflow errors may use audited native BGE input truncation; fallback batch counts and affected input texts are reported separately. In-range cached vectors are unchanged.",
        ],
    }
    write_json(campaign / "efficiency.json", record)
    columns = [
        "dataset", "questions", "em", "f1", "construction_all_attempts_s",
        "query_phase_wall_s", "query_mean_s", "query_p50_s", "query_p95_s",
        "retrieval_mean_s", "generation_mean_s", "index_MiB",
        "construction_llm_successes", "query_llm_successes",
        "input_tokens", "output_tokens", "failed_attempts",
        "local_response_cache_hits", "embedding_batch_requests",
        "entity_nodes", "hyperedge_nodes", "incidence_edges", "workers",
        "query_all_attempts_wall_s", "query_all_attempts_questions_per_s",
    ]
    with (campaign / "paper_efficiency.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            query = row["query_latency"]
            writer.writerow({
                "dataset": row["dataset"], "questions": row["questions"],
                "em": row["primary_em"], "f1": row["primary_f1"],
                "construction_all_attempts_s": row["construction_all_attempts_s"],
                "query_phase_wall_s": row["wall_time"].get("query_phase_wall_s"),
                "query_mean_s": query["end_to_end"]["mean_s"],
                "query_p50_s": query["end_to_end"]["p50_s"],
                "query_p95_s": query["end_to_end"]["p95_s"],
                "retrieval_mean_s": query["retrieval_including_query_extraction"]["mean_s"],
                "generation_mean_s": query["answer_generation"]["mean_s"],
                "index_MiB": row["storage"]["persistent_index_bytes"] / 1024 ** 2,
                "construction_llm_successes": row["api_by_pipeline_phase"]["construction"]["successful_calls"],
                "query_llm_successes": row["api_by_pipeline_phase"]["query"]["successful_calls"],
                "input_tokens": row["api"]["input_tokens"],
                "output_tokens": row["api"]["output_tokens"],
                "failed_attempts": row["api"]["failed_attempts"],
                "local_response_cache_hits": row["api"]["local_response_cache_hits"],
                "embedding_batch_requests": row["embedding"]["remote_loopback_batch_requests"],
                "entity_nodes": row["graph"]["entity_nodes"],
                "hyperedge_nodes": row["graph"]["hyperedge_nodes"],
                "incidence_edges": row["graph"]["incidence_edges"],
                "workers": row["concurrency"],
                "query_all_attempts_wall_s": row["query_all_attempts_wall_s"],
                "query_all_attempts_questions_per_s": row["query_all_attempts_questions_per_s"],
            })
    return record


def save_reproducibility(campaign, provenance, verified_rows):
    anchors = [
        "provenance.json", "prompts.json", "datasets.lock.json", "execution.json",
        "embedding_contract.json", "environment.json",
        "embedding_service.json",
        "source_snapshot/requirements.lock.txt", "tracked_source.patch",
        "recovery_manifest.json",
    ]
    write_json(campaign / "reproducibility.json", {
        "config": provenance["config"],
        "config_sha256": provenance["config_sha256"],
        "source_sha256": provenance["source_sha256"],
        "artifact_sha256": {
            path: file_hash(campaign / path) for path in anchors if (campaign / path).exists()},
        "sample_ids": {
            row["dataset"]: read_json(campaign / row["dataset"] / "data_manifest.json")["ids"]
            for row in verified_rows},
        "replay_note": (
            "Original request/response records enable exact response replay. Live API "
            "temperature zero is not a guarantee of identical future outputs or timings. "
            "Upstream asynchronous tie ordering can also affect evidence ordering."),
        "credentials": "Not stored; use the authorized shared environment launcher.",
    })
