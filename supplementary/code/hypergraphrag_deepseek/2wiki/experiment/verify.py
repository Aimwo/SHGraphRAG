"""Offline completion, efficiency and per-attempt evidence verification."""
from __future__ import annotations

from collections import Counter, defaultdict
import statistics

import numpy as np
import networkx as nx

from .common import (
    ROOT, file_hash, file_inventory, read_json, read_jsonl, signature, write_json,
)
from .evaluate import evaluate
from .prepare import validate_inputs
from .transport import provider_identity, usage_counts

from .config import load_config
DATASET = load_config(ready=False)["dataset_id"]


def usage_ledger(records):
    totals, known, missing = Counter(), Counter(), Counter()
    for record in records:
        for key, value in usage_counts(record.get("raw_usage")).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                totals[key] += value
                known[key] += 1
            else:
                missing[key] += 1
    return {
        key: {"known_sum": totals[key] if known[key] else None,
              "attempts_with_value": known[key], "attempts_without_value": missing[key]}
        for key in ("input_tokens", "output_tokens", "total_tokens",
                    "cached_input_tokens", "reasoning_output_tokens")}


def audit_requests(run, config):
    scope = signature(provider_identity(config))
    calls, attempt_ids = {}, set()
    for kind, filename in (("text", "api_calls.jsonl"), ("embedding", "embedding_calls.jsonl")):
        path = run / filename
        records = read_jsonl(path) if path.exists() else []
        for row in records:
            if row["attempt_id"] in attempt_ids:
                raise ValueError("Duplicate physical attempt ID")
            attempt_ids.add(row["attempt_id"])
            evidence = read_json(run / row["raw_artifact"])
            if evidence["record"] != row:
                raise ValueError("Physical attempt artifact/log mismatch")
            if signature({k: v for k, v in evidence.items() if k != "record"}) != row["raw_payload_sha256"]:
                raise ValueError("Physical attempt input/response hash mismatch")
            if (row["active_requests_on_send"] > config["max_in_flight"]
                    or row["active_embedding_requests_on_send"] > 4):
                raise ValueError("Request concurrency ceiling exceeded")
            endpoint = (config["base_url"].rstrip("/") + "/chat/completions" if kind == "text"
                        else config["embedding"]["base_url"] + "/api/embed")
            request = evidence["request"]
            if (row["endpoint"] != endpoint or evidence["endpoint"] != endpoint
                    or row["scope_sha256"] != scope
                    or row["request_sha256"] != signature({
                        "scope": scope, "endpoint": endpoint, "body": request})):
                raise ValueError("Request provider/configuration/input identity differs")
            if kind == "text":
                if request["model"] != config["model"] or row["requested_model"] != config["model"]:
                    raise ValueError("A physical request used a different model")
                if row["credential_id"] != config["credential_id"]:
                    raise ValueError("Mixed credential identities")
                if row["status"] != "failed" and row.get("returned_model") not in config["allowed_returned_models"]:
                    if row["status"] != "cancelled":
                        raise ValueError("A response used a different model")
            elif (len(request["input"]) > 32 or request["model"] != "bge-m3:latest"
                  or request.get("options") != config["embedding"]["options"] or "truncate" in request):
                raise ValueError("Embedding identity/batch/reference options differ")
        calls[kind] = records
    started = read_jsonl(run / "request_started.jsonl")
    if (len(started) != len(attempt_ids)
            or {r["attempt_id"] for r in started} != attempt_ids):
        raise ValueError("A started request lacks final evidence (possible hard interruption)")
    return calls


def verify(run=None):
    run = ROOT / "run" if run is None else run
    inputs = validate_inputs()
    if not (run / "index_complete.json").exists():
        raise ValueError("Index completion evidence is missing")
    ids = read_json(ROOT / "inputs/ids.json")
    predictions = [read_json(p) for p in (run / "predictions").glob("*.json")]
    if len(predictions) != 1000 or {p["id"] for p in predictions} != set(ids):
        raise ValueError("Every frozen question must have an explicit prediction outcome")
    contract = read_json(run / "provenance.json")
    if contract["config"]["max_in_flight"] != 100:
        raise ValueError("Run concurrency contract differs")
    for relative, expected in contract["source_sha256"].items():
        if (file_hash(run / "source_snapshot" / relative) != expected
                or file_hash(ROOT / relative) != expected):
            raise ValueError("Run/source snapshot hash changed: " + relative)
    index = read_json(run / "index_complete.json")
    if index["files"] != file_inventory(run / index["directory"]):
        raise ValueError("Completed index hashes changed")
    graph = nx.read_graphml(run / index["directory"] / "graph_chunk_entity_relation.graphml")
    roles = Counter(data.get("role") for _, data in graph.nodes(data=True))
    actual_graph = {
        "full_docs": len(read_json(run / index["directory"] / "kv_store_full_docs.json")),
        "chunks": len(read_json(run / index["directory"] / "kv_store_text_chunks.json")),
        "entity_nodes": roles["entity"], "hyperedge_nodes": roles["hyperedge"],
        "incidence_edges": graph.number_of_edges(),
        "nodes_without_role": sum(v for k, v in roles.items() if k not in ("entity", "hyperedge"))}
    if actual_graph != index["graph"]:
        raise ValueError("Graph statistics differ from the constructed index")
    from .run import validate_prediction
    queries = {q["id"]: q for q in read_jsonl(ROOT / "inputs/queries.jsonl")}
    passage_ids = {p["id"] for p in read_jsonl(ROOT / "inputs/candidate_corpus.jsonl")}
    query_attempts = read_jsonl(run / "query_attempts.jsonl")
    attempt_lookup = {}
    for row in query_attempts:
        if file_hash(run / row["artifact"]) != row["artifact_sha256"]:
            raise ValueError("Query attempt artifact hash changed")
        if row["query_attempt_id"] in attempt_lookup:
            raise ValueError("Duplicate query attempt ID")
        attempt_lookup[row["query_attempt_id"]] = row
    for prediction in predictions:
        validate_prediction(prediction, queries[prediction["id"]], contract["config"],
                            signature(contract), signature(index["files"]), passage_ids)
        row = attempt_lookup[prediction["query_attempt_id"]]
        if read_json(run / row["artifact"]) != prediction:
            raise ValueError("Latest prediction differs from immutable query attempt")
    calls = audit_requests(run, contract["config"])
    from .capacity_recovery import audit_capacity_import
    history, capacity_import = audit_capacity_import(
        run, contract.get("capacity_recovery_import"), contract["config"])
    history_ids = {r["attempt_id"] for items in history.values() for r in items}
    current_ids = {r["attempt_id"] for items in calls.values() for r in items}
    if history_ids & current_ids:
        raise ValueError("Historical physical attempts were counted twice")
    from .common import canonical
    history_path = run / "capacity_history_reporting_rows.jsonl"
    history_path.write_text("".join(canonical({**row,
        "_history_source_directory": row.get("_history_source_directory", capacity_import["source_directory"])}) + "\n"
        for items in history.values() for row in items), encoding="utf-8")
    for kind in calls:
        calls[kind] = history[kind] + calls[kind]
    if any(not records for records in calls.values()):
        raise ValueError("Both text and embedding task attempts are required")
    metrics = evaluate(run)
    if metrics["denominator"] != inputs["questions"]:
        raise ValueError("Metric denominator differs")
    construction = read_jsonl(run / "construction_attempts.jsonl")
    invocations = read_jsonl(run / "campaign_attempts.jsonl")
    phases = read_jsonl(run / "phase_events.jsonl")
    latencies = [p["latency_s"] for p in predictions]
    storage = defaultdict(int)
    for path in run.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(run)
        category = ("index" if relative.parts[0].startswith("index_") else relative.parts[0])
        storage[category] += path.stat().st_size
    by_stage = defaultdict(list)
    for row in calls["text"]:
        by_stage[row["stage"]].append(row)
    from .summary_policy import audit_summary_outputs
    summary_audit = audit_summary_outputs(run, graph, index["directory"])
    from .summary_reuse import audit_seeded_validation
    validation_audit = audit_seeded_validation(run)
    imported_timing = (capacity_import["timing"] if capacity_import is not None else {
        "construction_wall_s_including_observed_upper_bounds": 0.0,
        "invocation_wall_s_including_observed_upper_bounds": 0.0,
    })  # A fresh run has no imported physical history.
    report = {
        "status": "verified_complete", "dataset": DATASET, "questions": 1000,
        "capacity_recovery_import_audit": capacity_import,
        "summary_policy_audit": summary_audit,
        "preloaded_summary_validation_audit": validation_audit,
        "historical_physical_attempts_file": {
            "path": history_path.name, "sha256": file_hash(history_path)},
        "missing_or_failed_predictions": metrics["missing_or_failed"],
        "native_alias_em": metrics["native_alias_em"], "native_alias_f1": metrics["native_alias_f1"],
        "construction_wall_s_all_attempts": sum(a["wall_s"] for a in construction)
            + imported_timing["construction_wall_s_including_observed_upper_bounds"],
        "construction_last_success_wall_s": index["wall_s"],
        "execution_wall_s_all_invocations": sum(a["wall_s"] for a in invocations)
            + imported_timing["invocation_wall_s_including_observed_upper_bounds"],
        "query_phase_wall_s_all_attempts": sum(p["wall_s"] for p in phases if p["phase"] == "query_finished"),
        "latency_s": {"mean": statistics.mean(latencies),
                      "p50": float(np.percentile(latencies, 50)),
                      "p95": float(np.percentile(latencies, 95))},
        "latency_note": "Per-question observed latency includes request queue time, cache effects and shared-host contention.",
        "query_attempts_including_resumes": len(query_attempts),
        "query_attempt_latency_sum_s": sum(p["latency_s"] for p in query_attempts),
        "graph": index["graph"], "storage_bytes_by_directory": dict(storage),
        "active_index_storage_bytes": sum(p["bytes"] for p in index["files"].values()),
        "text_physical_attempts": len(calls["text"]),
        "embedding_physical_attempts": len(calls["embedding"]),
        "text_attempt_outcomes": dict(Counter(r["status"] for r in calls["text"])),
        "embedding_attempt_outcomes": dict(Counter(r["status"] for r in calls["embedding"])),
        "text_usage_all_attempts": usage_ledger(calls["text"]),
        "text_usage_by_stage": {k: usage_ledger(rows) for k, rows in by_stage.items()},
        "embedding_native_usage": [
            {"attempt_id": r["attempt_id"], **r.get("embedding_native_usage", {})}
            for r in calls["embedding"]],
        "path_question_repository": contract["method"]["path_question_repository"],
        "peak_combined_requests": max(r["active_requests_on_send"]
                                      for rows in calls.values() for r in rows),
        "peak_embedding_requests": max(r["active_embedding_requests_on_send"]
                                       for rows in calls.values() for r in rows),
        "text_cache_hits": len([
            r for r in read_jsonl(run / "cache_hits.jsonl") if r["kind"] == "text"]),
        "raw_inputs_and_responses": "one sanitized artifact for every physical attempt",
        "monetary_cost": None,
        "monetary_cost_note": "No verified relay tariff was provided; token counts are not a settled charge.",
        "embedding_runtime_digest": contract["config"]["embedding"]["actual_digest"],
        "embedding_runtime_digest_note": "Null means the actual runtime model digest was not verified.",
    }
    write_json(run / "verification.json", report)
    return report


if __name__ == "__main__":
    print(verify())
