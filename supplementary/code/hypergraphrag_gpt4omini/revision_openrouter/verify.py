"""Read-only independent completion/provenance checks and efficiency report."""
from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import json
from pathlib import Path
import statistics

import networkx as nx
import numpy as np

from .run import BASE, PROJECT, LOCK, document_text, parse_answer, read_json, read_jsonl
from .evaluate import evaluate_run
from .transport import (
    MODEL, PROVIDER, REASONING_EFFORT, RETURNED_MODELS,
    file_hash, signature, write_json,
)


def validate_response(saved, key):
    request = saved["request"]
    status = saved.get("status")
    valid_terminal = (
        status == "completed" and saved.get("finish_reason") == "stop"
        or status == "provider_filtered" and (
            saved.get("finish_reason") == "content_filter" or saved.get("refusal"))
        or status == "output_truncated" and saved.get("finish_reason") == "length"
        or status == "output_incomplete" and "finish_reason" in saved
        and saved["finish_reason"] is None)
    if (signature(request) != key or saved.get("request_sha256") != key
            or saved["model"] != MODEL or saved.get("provider") != PROVIDER
            or saved.get("returned_model") not in RETURNED_MODELS
            or not valid_terminal or request["model"] != MODEL
            or request.get("temperature") != 0 or "reasoning" in request):
        raise RuntimeError("Cached API response/request identity is invalid")
    if status in ("provider_filtered", "output_truncated", "output_incomplete"):
        response = saved.get("raw_response") or {}
        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        valid_original = (
            status == "provider_filtered"
            and (choice.get("finish_reason") == "content_filter" or message.get("refusal"))
            or status == "output_truncated" and choice.get("finish_reason") == "length"
            or status == "output_incomplete" and choice.get("finish_reason") is None
            and not message.get("refusal") and len(response.get("choices") or []) == 1)
        if (response.get("model") != saved["returned_model"]
                or choice.get("finish_reason") != saved["finish_reason"]
                or not valid_original):
            raise RuntimeError("Unavailable response lacks its original provider outcome")


def summarize(campaign):
    entries = {r["dataset"]: r for r in read_json(LOCK)["datasets"]}
    provenance = read_json(campaign / "provenance.json")
    for relative, expected in provenance["source_sha256"].items():
        snapshot = campaign / "source_snapshot" / relative
        path = snapshot if snapshot.exists() else BASE / relative
        if file_hash(path) != expected:
            raise RuntimeError("Recorded source changed: " + relative)
    if signature(provenance["config"]) != provenance["config_sha256"]:
        raise RuntimeError("Config signature mismatch")
    for path, key in ((campaign / "datasets.lock.json", "dataset_lock_sha256"),
                      (campaign / "source_snapshot/requirements.lock.txt",
                       "dependency_lock_sha256")):
        if path.exists() and file_hash(path) != provenance[key]:
            raise RuntimeError("Snapshot lockfile changed: " + str(path.name))
    calls = (read_jsonl(campaign / "api_calls.jsonl")
             if (campaign / "api_calls.jsonl").exists() else [])
    cache_hits = (read_jsonl(campaign / "cache_hits.jsonl")
                  if (campaign / "cache_hits.jsonl").exists() else [])
    embedding_calls = (read_jsonl(campaign / "embedding_calls.jsonl")
                       if (campaign / "embedding_calls.jsonl").exists() else [])
    checked_responses = set()
    for call in calls:
        if call["status"] not in ("completed", "provider_filtered", "output_truncated", "output_incomplete") or call["request_sha256"] in checked_responses:
            continue
        saved = read_json(campaign / "cache/responses" /
                          (call["request_sha256"] + ".json"))
        validate_response(saved, call["request_sha256"])
        checked_responses.add(call["request_sha256"])
    rows = []
    for dataset, entry in entries.items():
        root = campaign / dataset
        data = read_json(root / "data_manifest.json")
        predictions = [read_json(p) for p in (root / "predictions").glob("*.json")]
        ids = [p["id"] for p in predictions]
        if len(ids) != len(set(ids)) or set(ids) != set(data["ids"]):
            raise RuntimeError("Missing, extra or duplicate predictions: " + dataset)
        if any(p["model"] != MODEL or p.get("provider") != PROVIDER
               or p["reasoning_effort"] != REASONING_EFFORT
               or p["status"] != "completed" for p in predictions):
            raise RuntimeError("Invalid prediction model/status: " + dataset)
        mapping = read_json(root / "document_passage_map.json")
        for prediction in predictions:
            if set(prediction["retrieved_passage_ids"]) - set(mapping.values()):
                raise RuntimeError("Retrieval provenance outside frozen corpus")
            if signature(prediction["context"]) != prediction["context_sha256"]:
                raise RuntimeError("Evidence context hash mismatch")
            if prediction.get("answer_outcome") in ("provider_filtered", "output_truncated", "output_incomplete"):
                refusal = prediction.get("unavailable_output") or prediction.get("provider_refusal") or {}
                saved = read_json(campaign / "cache/responses" /
                                  (refusal["request_sha256"] + ".json"))
                validate_response(saved, refusal["request_sha256"])
                if saved["status"] != prediction["answer_outcome"] or prediction["answer"] != "":
                    raise RuntimeError("Unavailable query must remain an empty scored answer")
            if prediction.get("answer_outcome") == "invalid_answer_schema":
                if prediction["answer"] != "":
                    raise RuntimeError("Malformed generated answer must remain empty")
                try:
                    parse_answer(prediction["raw_answer"])
                except ValueError:
                    pass
                else:
                    raise RuntimeError("Invalid-schema outcome contains a valid answer")
            if provenance["config"].get("telemetry", {}).get("query_subphase_timing"):
                components = [prediction[key] for key in (
                    "latency_s", "retrieval_latency_s", "answer_generation_latency_s")]
                if any(not np.isfinite(value) or value < 0 for value in components):
                    raise RuntimeError("Invalid per-query latency measurement")
                if components[0] + 0.02 < sum(components[1:]):
                    raise RuntimeError("Query phase measurements exceed end-to-end time")
        index_info = read_json(root / "index_complete.json")
        index_root = root / index_info["directory"]
        documents = read_json(index_root / "kv_store_full_docs.json")
        chunks = read_json(index_root / "kv_store_text_chunks.json")
        if set(documents) != set(mapping) or len(documents) != data["passage_count"]:
            raise RuntimeError("Index full-document coverage mismatch")
        corpus_source = PROJECT / entry["sources"]["candidate_corpus"]["path"]
        if file_hash(corpus_source) != data["corpus_sha256"]:
            raise RuntimeError("Frozen corpus changed")
        corpus = {p["passage_id"]: p for p in read_jsonl(corpus_source)}
        if any(documents[doc_id]["content"] != document_text(corpus[pid]).strip()
               for doc_id, pid in mapping.items()):
            raise RuntimeError("Indexed document content differs from frozen corpus")
        if len(chunks) != index_info["chunks"]:
            raise RuntimeError("Index chunk coverage mismatch")
        if provenance["config"].get("index_extraction_call_provenance"):
            processed = {
                record["chunk_id"] for record in calls + cache_hits
                if record.get("dataset") == dataset and record.get("stage") == "extract"
                and record.get("status", "completed") in (
                    "completed", "provider_filtered", "output_truncated", "output_incomplete")
                and record.get("chunk_id")
            }
            if set(chunks) - processed:
                raise RuntimeError("Missing completed extraction-call provenance")
        graph = nx.read_graphml(index_root / "graph_chunk_entity_relation.graphml")
        entities = {node for node, info in graph.nodes(data=True) if info.get("role") == "entity"}
        hyperedges = {node for node, info in graph.nodes(data=True) if info.get("role") == "hyperedge"}
        if len(entities | hyperedges) != graph.number_of_nodes():
            raise RuntimeError("Unknown graph roles")
        if any((a in entities) == (b in entities) for a, b in graph.edges()):
            raise RuntimeError("Graph contains a non-incidence edge")
        missing_sources = set()
        for _, node in graph.nodes(data=True):
            missing_sources.update(set(node["source_id"].split("<SEP>")) - set(chunks))
        if missing_sources:
            raise RuntimeError("Graph source IDs missing from chunk store")
        metrics = evaluate_run(root, PROJECT, entry)
        relevant = [r for r in calls if r.get("dataset") == dataset]
        tokens, stages = Counter(), defaultdict(Counter)
        for call in relevant:
            if (call["model"] != MODEL or call.get("provider") != PROVIDER
                    or call["reasoning_effort"] != REASONING_EFFORT
                    or (call["status"] == "completed"
                        and call.get("returned_model") not in RETURNED_MODELS)):
                raise RuntimeError("API record model/settings mismatch")
            tokens["attempts"] += 1
            tokens["completed"] += call["status"] == "completed"
            tokens["failed"] += call["status"] != "completed"
            tokens["provider_filtered_responses"] += (
                call["status"] == "provider_filtered"
                or call.get("error") in ("completion_content_filter", "completion_refusal"))
            tokens["truncated_output_responses"] += (
                call["status"] == "output_truncated" or call.get("error") == "completion_length")
            tokens["incomplete_output_responses"] += (
                call["status"] == "output_incomplete"
                or call.get("error") in ("completion_None", "completion_missing_finish_reason"))
            usage = call.get("usage", {})
            if call["status"] != "completed" and not usage:
                tokens["failed_attempts_without_reported_usage"] += 1
            for key in ("input_tokens", "output_tokens", "total_tokens"):
                tokens[key] += usage.get(key, 0) or 0
            tokens["cached_input_tokens"] += (usage.get("input_tokens_details") or {}).get("cached_tokens", 0) or 0
            reasoning_tokens = (usage.get("output_tokens_details") or {}).get("reasoning_tokens")
            if reasoning_tokens is not None:
                tokens["calls_with_reasoning_usage"] += 1
                tokens["reasoning_tokens"] += reasoning_tokens
                if reasoning_tokens:
                    raise RuntimeError("Provider reported reasoning tokens for GPT-4o-mini")
            stages[call["stage"]]["attempts"] += 1
            stages[call["stage"]]["input_tokens"] += usage.get("input_tokens", 0) or 0
            stages[call["stage"]]["output_tokens"] += usage.get("output_tokens", 0) or 0
        cached_usage = Counter()
        relevant_hits = [r for r in cache_hits if r.get("dataset") == dataset]
        for hit in relevant_hits:
            saved = read_json(campaign / "cache/responses" /
                              (hit["request_sha256"] + ".json"))
            validate_response(saved, hit["request_sha256"])
            for key in ("input_tokens", "output_tokens", "total_tokens"):
                cached_usage[key] += saved.get("usage", {}).get(key, 0) or 0
        latencies = [p["latency_s"] for p in predictions]
        sizes = {str(p.relative_to(index_root)): p.stat().st_size
                 for p in index_root.rglob("*") if p.is_file()}
        attempts_path = root / "construction_attempts.jsonl"
        attempts = read_jsonl(attempts_path) if attempts_path.exists() else []
        refused_records = [
            record for record in relevant + relevant_hits
            if record.get("status") == "provider_filtered"
            or record.get("error") in ("completion_content_filter", "completion_refusal")]
        truncated_records = [
            record for record in relevant + relevant_hits
            if record.get("status") == "output_truncated"
            or record.get("error") == "completion_length"]
        incomplete_records = [
            record for record in relevant + relevant_hits
            if record.get("status") == "output_incomplete"
            or record.get("error") in ("completion_None", "completion_missing_finish_reason")]
        row = {
            **metrics, "graph": index_info,
            "index_storage_bytes": sum(sizes.values()), "index_files_bytes": sizes,
            "api": dict(tokens), "api_by_stage": {k: dict(v) for k, v in stages.items()},
            "cached_calls": len(relevant_hits),
            "cached_response_usage_from_prior_paid_work": dict(cached_usage),
            "logical_llm_calls_including_cache": tokens["completed"] + len(relevant_hits),
            "embedding_requests": sum(r.get("dataset") == dataset for r in embedding_calls),
            "query_latency": {"mean_s": statistics.mean(latencies),
                              "p50_s": float(np.percentile(latencies, 50)),
                              "p95_s": float(np.percentile(latencies, 95))},
            "query_count_with_cached_calls": sum(p["cached_llm_calls"] > 0 for p in predictions),
            "construction_attempts": attempts,
            "total_construction_attempt_elapsed_s": (
                sum(r["elapsed_s"] for r in attempts) if attempts else index_info["elapsed_s"]),
            "concurrent_query_workers": provenance["workers"],
            "provider_refusals": {
                "unique_requests": len({r["request_sha256"] for r in refused_records}),
                "initially_filtered_chunks": len({
                    r["chunk_id"] for r in refused_records
                    if r["stage"] == "extract" and r.get("chunk_id")}),
                "empty_answer_questions": sum(
                    p.get("answer_outcome") == "provider_filtered" for p in predictions),
                "policy": provenance["config"].get("provider_refusal_policy"),
            },
            "output_truncations": {
                "unique_requests": len({r["request_sha256"] for r in truncated_records}),
                "initially_truncated_chunks": len({
                    r["chunk_id"] for r in truncated_records
                    if r["stage"] == "extract" and r.get("chunk_id")}),
                "empty_answer_questions": sum(
                    p.get("answer_outcome") == "output_truncated" for p in predictions),
                "policy": provenance["config"].get("output_truncation_policy"),
            },
            "incomplete_outputs": {
                "unique_requests": len({r["request_sha256"] for r in incomplete_records}),
                "initially_incomplete_chunks": len({
                    r["chunk_id"] for r in incomplete_records
                    if r["stage"] == "extract" and r.get("chunk_id")}),
                "empty_answer_questions": sum(
                    p.get("answer_outcome") == "output_incomplete" for p in predictions),
                "policy": provenance["config"].get("output_incomplete_policy"),
            },
            "invalid_answer_schema_questions": sum(
                p.get("answer_outcome") == "invalid_answer_schema" for p in predictions),
        }
        rows.append(row)
    full = all(row["expected"] == 1000 and row["completed"] == 1000 for row in rows)
    report = {"status": "verified_complete" if full else "verified_pilot",
              "full_benchmark": full, "datasets": rows,
              "verifier_sha256": file_hash(Path(__file__)),
              "config_sha256": provenance["config_sha256"],
              "average_em": statistics.mean(r["primary_em"] for r in rows),
              "average_f1": statistics.mean(r["primary_f1"] for r in rows),
              "average_native_em": statistics.mean(r["native_alias_em"] for r in rows),
              "average_native_f1": statistics.mean(r["native_alias_f1"] for r in rows),
              "limitations": [
                  "New frozen development samples and pooled candidate corpus, not full Wikipedia.",
                  "LLM/embedding substitution; not a claim to match original paper scores.",
                  "Shared remote API/local GPU load affects wall-clock latency.",
                  "Construction/query times include recorded response and embedding cache reuse.",
                  "Source-section ordering defines passage recall diagnostics; no gold reranking.",
                  "No cost in currency inferred without a verified provider price.",
                  "Timeout attempts may incur unreported server-side tokens; reported usage is a lower bound.",
              ]}
    from .efficiency import build_efficiency, save_reproducibility
    efficiency = build_efficiency(campaign, rows)
    save_reproducibility(campaign, provenance, rows)
    lines = ["# HyperGraphRAG verified results", "", f"Status: `{report['status']}`.", "",
             "Model: OpenRouter `openai/gpt-4o-mini`, temperature 0; embedding: pinned local BGE-M3.",
             "", "| Dataset | Questions | EM | F1 | Native EM | Native F1 |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    lines += [f"| {r['dataset']} | {r['completed']}/{r['expected']} | "
              f"{r['primary_em']:.2f} | {r['primary_f1']:.2f} | "
              f"{r['native_alias_em']:.2f} | {r['native_alias_f1']:.2f} |" for r in rows]
    lines += ["", "Primary EM/F1 match the independent SHGraphRAG run's documented",
              "single-reference Hotpot-style protocol. Native columns additionally",
              "apply dataset-native aliases and MuSiQue empty-answer semantics.", "",
              "See `verification.json` for coverage, incidence-graph checks, token",
              "counts, cache reuse, index storage, construction time and latency.", "",
              *["- " + item for item in report["limitations"]], ""]
    lines += [
        "## Efficiency measurements", "",
        "| Dataset | Build, all attempts (s) | Query mean (s) | Query P95 (s) | Index (MiB) | LLM attempts | Input tokens | Output tokens |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in efficiency["datasets"]:
        lines.append(
            f"| {row['dataset']} | {row['construction_all_attempts_s']:.2f} | "
            f"{row['query_latency']['end_to_end']['mean_s']:.3f} | "
            f"{row['query_latency']['end_to_end']['p95_s']:.3f} | "
            f"{row['storage']['persistent_index_bytes']/1024**2:.2f} | "
            f"{row['api']['attempts']} | {row['api']['input_tokens']} | "
            f"{row['api']['output_tokens']} |")
    lines += [
        "", efficiency["latency_definition"], "",
        "Detailed phase timings, retrieval/generation splits, construction/query "
        "LLM counts, cache hits, retries, embedding requests, raw provider-reported "
        "costs and resource observations are in `efficiency.json`. "
        "`paper_efficiency.csv` is a paper-table export.", "",
        "Query wall time across all attempts, including interrupted work before "
        "a resume, is `query_all_attempts_wall_s`; its throughput uses all saved "
        "questions. `wall_time.query_phase_wall_s` covers the last invocation "
        "only. Failed query phases use their recorded phase-start and invocation-"
        "exit timestamps; completed phases use monotonic elapsed time. Recovery "
        "downtime is excluded. Imported completed answers retain their original "
        "contexts and per-question retrieval/generation latencies.", "",
        "Local embedding HTTP failures are retried at most four times with the "
        "exact same payload. Every newly instrumented HTTP attempt is retained "
        "in `embedding_http_attempts.jsonl`, with raw failure responses under "
        "`cache/embedding_failures/`. Earlier successful batches and failed "
        "construction wall times are preserved; the first HTTP 400 response "
        "body was not captured and its server-side cause remains unknown.", "",
        "The final recovery uses a dedicated local Ollama process at "
        "`http://127.0.0.1:11436/api/embed` with per-request `num_ctx=8192` "
        "and `num_batch=8192`, the same pinned BGE weights and a primary "
        "`truncate=false` request. An exact HTTP 400 context-overflow response "
        "permits a separately recorded `truncate=true` fallback with the same "
        "model, full input text and execution options. Ollama then truncates "
        "only the overlength input to its context window. Original failures, "
        "fallback payloads, responses and hashes are preserved under "
        "`cache/embedding_overflows/`; diagnostic vectors are never imported. "
        "Previously successful in-range vectors remain unchanged. "
        "This avoids competing runtime options forcing reloads on the shared "
        "default service. Runtime/executable details are in "
        "`embedding_service.json` and the embedding contract; earlier shared "
        "service attempts remain in the combined histories.", "",
        "Path-question repository: not applicable to official HyperGraphRAG; "
        "it does not construct this SHGraphRAG-specific component.", "",
        "Reproduction parameters, prompts, dependency versions, code snapshots, "
        "dataset hashes and every sampled ID are anchored by `reproducibility.json`.",
        "",
    ]
    if provenance["config"].get("embedding_overflow_policy"):
        lines += [
            "## Embedding context overflows", "",
            "| Dataset | Successful fallback batches | Input texts in fallback batches |",
            "| --- | ---: | ---: |",
            *[
                f"| {row['dataset']} | "
                f"{row['embedding']['overflow_fallback_completed_batches']} | "
                f"{row['embedding']['overflow_fallback_input_texts']} |"
                for row in efficiency["datasets"]
            ],
            "",
            "Fallback batch text counts include every input resubmitted in a "
            "batch. They do not establish that each text exceeded the context "
            "window. Exact overflowing text counts are only known for "
            "single-input batches. This embedding input policy is a disclosed "
            "implementation substitution, with no changes to text-model "
            "prompts or output limits.", "",
        ]
    if provenance["config"].get("text_error_recovery"):
        lines += [
            "## Structured provider errors and bounded retries", "",
            "Explicit provider availability errors inside an HTTP-200 response "
            "use the same bounded transient status-code rules as HTTP errors: "
            "at most four exact-payload attempts per request invocation. "
            "Model, messages, temperature, output cap and routing are unchanged "
            "between attempts. The partial failed output is never used. "
            "Instrumented failures preserve raw responses in cache/text_failures "
            "and retain every reported usage/cost record, invocation ID, attempt "
            "number and retry delay in api_calls.jsonl. Prior captured failures "
            "remain unchanged in the inherited log and raw-failure archive. "
            "All failed construction attempts remain included in timing totals. "
            "Filter/refusal, length-limit and missing-finish outcomes retain "
            "their separate terminal policies. Billing/authentication failures "
            "require attention and are not retried as availability errors.", "",
        ]
    if provenance["config"].get("provider_refusal_policy"):
        lines += [
            "## Provider-filtered outputs", "",
            "Explicit provider content filters/refusals are terminal unavailable "
            "outputs, recorded with the original response and usage. They are "
            "not retried or replaced with another model/provider/prompt. "
            "All corpus documents/chunks and all 1000 question IDs per dataset "
            "remain in scope. Filtered initial extraction adds no graph facts; "
            "filtered later extraction stops that chunk's generation and retains "
            "earlier complete facts. Filtered summaries retain existing descriptions. "
            "Filtered queries/answers receive empty answers in the full denominator. "
            "Partial filtered text is never used as evidence or an answer.", "",
            "| Dataset | Unique filtered requests | Initial filtered chunks | Empty-answer questions from filtering |",
            "| --- | ---: | ---: | ---: |",
        ]
        lines += [
            f"| {row['dataset']} | {row['provider_refusals']['unique_requests']} | "
            f"{row['provider_refusals']['initially_filtered_chunks']} | "
            f"{row['provider_refusals']['empty_answer_questions']} |" for row in rows]
        lines.append("")
    if provenance["config"].get("output_incomplete_policy"):
        lines += [
            "## Outputs without completion boundaries", "",
            "Missing/null finish reasons are recorded separately as output_incomplete. "
            "The cause is unknown; they are not classified as token-cap termination "
            "or explicit filtering. Raw responses and original usage remain preserved "
            "without another paid request or partial-text use. Initial extraction "
            "retains the corpus/chunk/vector without adding facts from this response. "
            "Later extraction retains earlier complete facts, summaries retain existing "
            "descriptions, and query/answer failures receive empty scored answers. "
            "All frozen question IDs remain in the denominator.", "",
            "| Dataset | Unique incomplete requests | Initial incomplete chunks | Empty answers from incomplete output |",
            "| --- | ---: | ---: | ---: |",
        ]
        lines += [
            f"| {row['dataset']} | {row['incomplete_outputs']['unique_requests']} | "
            f"{row['incomplete_outputs']['initially_incomplete_chunks']} | "
            f"{row['incomplete_outputs']['empty_answer_questions']} |" for row in rows]
        lines.append("")
    if provenance["config"].get("output_truncation_policy"):
        lines += [
            "## Truncated and malformed outputs", "",
            "Outputs with finish_reason=length are terminal unavailable outputs "
            "under unchanged token caps. They are not retried or parsed for "
            "partial facts/answers. Corpus documents and question IDs remain "
            "in scope. Previously completed extraction facts are retained; "
            "unavailable summaries retain existing descriptions. Truncated "
            "queries/answers and malformed answer JSON receive empty scored "
            "answers in the full denominator. Original raw outputs and usage "
            "are retained.", "",
            "| Dataset | Unique truncated requests | Initial truncated chunks | Empty answers from truncation | Malformed answers |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        lines += [
            f"| {row['dataset']} | {row['output_truncations']['unique_requests']} | "
            f"{row['output_truncations']['initially_truncated_chunks']} | "
            f"{row['output_truncations']['empty_answer_questions']} | "
            f"{row['invalid_answer_schema_questions']} |" for row in rows]
        lines.append("")
    (campaign / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(campaign / "verification.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    args = parser.parse_args()
    report = summarize(BASE / "revision_openrouter/runs" / args.run)
    print(json.dumps({k: v for k, v in report.items() if k != "datasets"}, ensure_ascii=True))
