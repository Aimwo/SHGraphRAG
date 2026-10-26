"""Check completed run evidence offline; never call a model or rebuild a graph."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import zipfile

from reproduction.campaign import DEFAULT_RUN, read_records, usage_summary
from reproduction.freeze_benchmark import DESTINATION, ROOT, sha256
from reproduction.runtime import completed_interval_peak, utc_now, write_json
from reproduction.relay_config import load_relay_config, config_identity
from config.settings import EMBEDDING_MAX_IN_FLIGHT, SAMPLE_SIZE, TEXT_MAX_WORKERS


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    root = args.run_dir
    relay = load_relay_config()
    checks = []

    def check(name, passed, evidence=None):
        checks.append({"check": name, "passed": bool(passed), "evidence": evidence})

    state = read(root / "state.json")
    config = read(root / "effective_config.json")
    summary = read(root / "final_summary.json")
    construction = read(root / "construction_summary.json")
    metrics = read(root / "metrics.json")
    predictions = read(root / "qa/predictions_with_supporting_facts.json")
    answers = read(root / "qa/predictions.json")["answer"]
    ids = read(DESTINATION / "ids.json")
    benchmark = read(Path(os.environ.get("SHGRAPH_BENCHMARK_CONFIG",
        ROOT / "reproduction/benchmark.json")))
    if benchmark.get("aliases"):
        check("native_alias_hash", sha256(ROOT / benchmark["aliases"]) == benchmark["aliases_sha256"])
    manifest = read(DESTINATION / "manifest.json")
    check("controller_finished", state["status"] == "completed"
          and state["full_reproduction_completed"])
    check("fixed_sample", len(ids) == len(set(ids)) == SAMPLE_SIZE
          and config["sampling_seed"] == manifest["seed"] == 42
          and sha256(DESTINATION / "ids.json") == config["sample_ids_sha256"])
    check("sample_manifest_unchanged",
          sha256(DESTINATION / "manifest.json") == config["sample_manifest_sha256"])
    for name, info in manifest["files"].items():
        check(f"frozen_artifact_hash:{name}",
              sha256(DESTINATION / name) == info["sha256"])
    check("prediction_id_order", [row["id"] for row in predictions] == ids)
    check("prediction_coverage", set(answers) == set(ids)
          and len(predictions) == summary["questions"] == SAMPLE_SIZE)
    check("successful_question_checkpoints",
          all(row["status"] == "completed" for row in predictions))
    check("answer_types_and_checkpoint_handoff",
          all(isinstance(row["answer"], str)
              and answers[row["id"]] == row["answer"] for row in predictions))
    check("metric_coverage", metrics["denominator"] == metrics["prediction_count"] == SAMPLE_SIZE
          and not metrics["missing_prediction_ids"]
          and [row["id"] for row in metrics["per_question"]] == ids)
    for metric in ("answer_em", "answer_f1"):
        mean = sum(row[metric] for row in metrics["per_question"]) / SAMPLE_SIZE
        check(f"metric_aggregate:{metric}", abs(mean - metrics[metric]) < 1e-12, mean)
    check("candidate_graph_alignment",
          construction["alignment"]["passed"]
          and construction["alignment"]["documents"] == SAMPLE_SIZE)
    graph = construction["graph"]
    online = {
        row["name"] for row in graph["vector_indexes"] if row["state"] == "ONLINE"
    }
    check("constructed_graph_and_indexes",
          graph["documents"] == SAMPLE_SIZE and graph["entities"] > 0
          and {"entity", "chunk"} <= online)

    code_hash = config["source_bundle_sha256"]
    source_hashes = read(root / "source" / f"{code_hash}.hashes.json")
    actual_manifest_hash = hashlib.sha256(
        json.dumps(source_hashes, sort_keys=True).encode()
    ).hexdigest()
    check("execution_source_manifest_hash", actual_manifest_hash == code_hash)
    source_archive = root / "source" / f"{code_hash}.zip"
    with zipfile.ZipFile(source_archive) as archive:
        failures = [
            name for name, expected in source_hashes.items()
            if hashlib.sha256(archive.read(name)).hexdigest() != expected
        ]
    check("execution_source_archive_contents", not failures, failures)
    changed_python = [
        name for name, expected in source_hashes.items()
        if name.endswith(".py") and sha256(ROOT / name) != expected
    ]
    check("archived_execution_python_unchanged", not changed_python, changed_python)

    records = read_records(root)
    physical = [row for row in records if row.get("api_request_started")]
    usage = usage_summary(root)
    pending = [row["id"] for row in records
               if row["status"] in ("waiting_for_slot", "in_flight")]
    check("no_outstanding_request_records", not pending, pending)
    check("physical_requests_have_known_final_status",
          all(row["status"] in ("completed", "failed", "refused") for row in physical))
    check("provider_credential_model_identity",
          config["relay_config_sha256"] == config_identity(relay)
          and all(row["requested_model"] == relay["model"]
                  and row["credential_id"] == relay["credential_id"]
                  and row["relay_config_sha256"] == config_identity(relay)
                  for row in physical))
    completed = [row for row in physical if row["status"] == "completed"]
    check("completed_returned_model_and_status",
          all(row["returned_model"] in relay["allowed_returned_models"]
              and row["response_status"] == "completed"
              and row["finish_reason"] == "stop"
              and row["http_status"] == 200 for row in completed))
    refused = [row for row in physical if row["status"] == "refused"]
    check("explicit_refusals_are_not_normal_answers", all(
        row["returned_model"] in relay["allowed_returned_models"]
        and row["response_status"] == "refused"
        and row["finish_reason"] == "content_filter" and row["http_status"] == 200
        for row in refused
    ))
    check("text_concurrency_cap",
          usage["observed_completed_interval_peak_concurrency"] <= TEXT_MAX_WORKERS,
          usage["observed_completed_interval_peak_concurrency"])
    embeddings = [
        json.loads(line)
        for line in (root / "embedding_calls.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    embedding_peak = completed_interval_peak(embeddings)
    check("embedding_concurrency_cap",
          bool(embeddings)
          and config["engineering_scheduling"]["local_embedding_max_in_flight"] == EMBEDDING_MAX_IN_FLIGHT
          and all(row["local_gate_capacity"] == EMBEDDING_MAX_IN_FLIGHT
                  for row in embeddings)
          and 0 < embedding_peak <= EMBEDDING_MAX_IN_FLIGHT,
          {"configured_capacity": EMBEDDING_MAX_IN_FLIGHT,
           "observed_completed_interval_peak_concurrency": embedding_peak})
    payload_failures = []
    usage_failures = []
    for row in physical:
        payload = read(root / "requests" / f"{row['id']}.input.json")
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        if not (
            hashlib.sha256(encoded.encode("utf-8")).hexdigest() == row["request_sha256"]
            and payload["model"] == relay["model"]
            and payload["stream"] is False
            and "temperature" not in payload and "max_tokens" not in payload
            and isinstance(payload.get("messages"), list)
            and all(payload.get(key) == value
                    for key, value in relay["generation_parameters"].items())
        ):
            payload_failures.append(row["id"])
        tokens = row.get("usage")
        if tokens is not None:
            basic = [tokens.get(key) for key in
                     ("input_tokens", "output_tokens", "total_tokens")]
            cached = (tokens.get("input_tokens_details") or {}).get("cached_tokens", 0)
            reasoning = (tokens.get("output_tokens_details") or {}).get(
                "reasoning_tokens", 0
            )
            if not (
                all(isinstance(value, int) and value >= 0 for value in basic)
                and basic[0] + basic[1] == basic[2]
                and 0 <= cached <= basic[0] and 0 <= reasoning <= basic[1]
            ):
                usage_failures.append(row["id"])
    check("actual_payload_hashes_and_parameters", not payload_failures, payload_failures)
    check("provider_usage_arithmetic", not usage_failures, usage_failures)
    check("final_usage_matches_request_records",
          usage == read(root / "usage_summary.json"))
    ended_missing_usage = [
        row["id"] for row in physical if row.get("usage") is None
    ]
    report = {
        "generated_at": utc_now(),
        "passed": all(row["passed"] for row in checks),
        "new_model_api_requests": 0,
        "checks": checks,
        "execution_source_content_manifest_sha256": code_hash,
        "execution_source_archive_file_sha256": sha256(source_archive),
        "physical_request_statuses": dict(Counter(row["status"] for row in physical)),
        "ended_requests_without_provider_usage": ended_missing_usage,
        "empty_string_answers": sum(not row["answer"].strip() for row in predictions),
        "note": (
            "Checks verify saved execution evidence. API transport success does not "
            "prove semantic extraction or answer correctness. Unknown usage is not invented."
        ),
    }
    write_json(root / "completion_verification.json", report)
    print(json.dumps({
        "passed": report["passed"], "checks": len(checks),
        "failed_checks": [row for row in checks if not row["passed"]],
        "physical_request_statuses": report["physical_request_statuses"],
        "ended_requests_without_provider_usage": len(ended_missing_usage),
        "empty_string_answers": report["empty_string_answers"],
    }, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
