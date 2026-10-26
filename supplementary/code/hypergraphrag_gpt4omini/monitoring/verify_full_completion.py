"""Independent final acceptance audit; never imported by inference."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from revision_openrouter.transport import file_hash, signature, write_json
from revision_openrouter.embedding_client import OVERFLOW_ERROR
from monitoring.embedding_history import (
    EmbeddingHistoryError, current_contract, identity_matches,
    load_authenticated_embedding_history, request_options_match,
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def has_original_prefix(combined, original):
    if not combined.is_file() or not original.is_file():
        return False
    with combined.open("rb") as current, original.open("rb") as preserved:
        while chunk := preserved.read(1024 * 1024):
            if current.read(len(chunk)) != chunk:
                return False
    return True


def _safe_artifact_path(campaign, relative):
    if not isinstance(relative, str):
        return None
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        return None
    return campaign / path


def audit_embedding_http_records(campaign, dataset, records, config, *, history=None):
    """Validate normal embedding attempts and the narrow overflow fallback."""
    history_errors = []
    if history is None and (campaign / "provenance.json").is_file():
        try:
            history = load_authenticated_embedding_history(campaign, config)
        except (EmbeddingHistoryError, OSError, KeyError, ValueError) as exc:
            history_errors.append(str(exc))
    standalone = current_contract(config)

    def contract_for(record):
        if history is not None:
            return history.contracts.get(signature(record))
        return None if history_errors else standalone

    if history is not None:
        supplied = Counter(signature(record) for record in records)
        if any(count > history.occurrences[key] for key, count in supplied.items()):
            history_errors.append("Embedding rows absent from authenticated history or duplicated")
    invalid_identity = []
    invalid_truncate = []
    artifact_errors = []
    fallback_groups = {}
    original_overflow_attempts = {}

    for record in records:
        contract = contract_for(record)
        expected = (contract or standalone)["identity"]
        identity_ok = identity_matches(record, contract)
        identity_ok = identity_ok and (
            dataset is None or record.get("dataset") == dataset)
        if not identity_ok:
            invalid_identity.append(record.get("request_sha256"))

        truncate = record.get("truncate")
        if truncate is False:
            pass
        elif truncate is True and record.get("overflow_artifact"):
            fallback_groups.setdefault(
                record["overflow_artifact"], []).append(record)
        else:
            invalid_truncate.append(record.get("request_sha256"))

        failure_artifact = record.get("failure_artifact")
        failure_saved = None
        if failure_artifact:
            path = _safe_artifact_path(campaign, failure_artifact)
            if path is None or not path.is_file():
                artifact_errors.append(failure_artifact)
                continue
            failure_saved = read(path)
            request = failure_saved.get("request") or {}
            if (failure_saved.get("request_sha256") != record.get("request_sha256")
                    or signature(request) != record.get("request_sha256")
                    or failure_saved.get("http_status") != record.get("http_status")
                    or request.get("model") != expected["model"]
                    or request.get("truncate") != truncate
                    or not request_options_match(request, contract)):
                artifact_errors.append(failure_artifact)
            if (record.get("failure_artifact_sha256")
                    and record["failure_artifact_sha256"] != file_hash(path)):
                artifact_errors.append(failure_artifact)

        if (truncate is False and record.get("http_status") == 400
                and record.get("error") == "http_400"
                and (failure_saved or {}).get("response") == {"error": OVERFLOW_ERROR}):
            original_overflow_attempts.setdefault(
                record.get("request_sha256"), []).append(record)

    fallback_errors = []
    fallback_retry_count = 0
    completed_fallbacks = 0
    referenced_original_hashes = set()
    for artifact_name, rows in fallback_groups.items():
        fallback_retry_count += max(0, len(rows) - 1)
        rows = sorted(rows, key=lambda row: row.get("attempt", 0))
        artifact_path = _safe_artifact_path(campaign, artifact_name)
        if artifact_path is None or not artifact_path.is_file():
            fallback_errors.append(artifact_name)
            continue
        evidence = read(artifact_path)
        original = evidence.get("original_request") or {}
        fallback = evidence.get("fallback_request") or {}
        original_hash = evidence.get("original_request_sha256")
        fallback_hash = evidence.get("fallback_request_sha256")
        original_artifact_name = evidence.get("original_failure_artifact")
        original_rows = original_overflow_attempts.get(original_hash, [])
        referenced_original_hashes.add(original_hash)
        contract = contract_for(rows[0])
        expected = (contract or standalone)["identity"]
        policy = (contract or {}).get("overflow_policy")

        valid = [
            evidence.get("diagnostic_only") is False,
            evidence.get("policy") == policy,
            bool(policy),
            signature(original) == original_hash,
            signature(fallback) == fallback_hash,
            original.get("model") == expected["model"],
            original.get("truncate") is False,
            request_options_match(original, contract),
            fallback.get("model") == expected["model"],
            fallback.get("truncate") is True,
            request_options_match(fallback, contract),
            original.get("input") == fallback.get("input"),
            fallback == {**original, "truncate": True},
            len(rows) <= 4,
            [row.get("attempt") for row in rows]
            == list(range(1, len(rows) + 1)),
            all(row.get("truncate") is True
                and row.get("overflow_artifact") == artifact_name
                and row.get("request_sha256") == fallback_hash
                and (row.get("attempt") < 4
                     or row.get("retry_scheduled") is not True)
                for row in rows),
            all(contract_for(row) == contract
                and identity_matches(row, contract)
                for row in rows),
            bool(original_rows),
        ]
        original_artifact_path = _safe_artifact_path(
            campaign, original_artifact_name)
        if original_artifact_path is None or not original_artifact_path.is_file():
            valid.append(False)
        else:
            original_saved = read(original_artifact_path)
            valid.extend([
                original_artifact_name in {
                    row.get("failure_artifact") for row in original_rows},
                any(row.get("failure_artifact") == original_artifact_name
                    and contract_for(row) == contract
                    and identity_matches(row, contract)
                    for row in original_rows),
                original_saved.get("request_sha256") == original_hash,
                signature(original_saved.get("request") or {}) == original_hash,
                original_saved.get("http_status") == 400,
                (original_saved.get("response") or {})
                == {"error": OVERFLOW_ERROR},
            ])
        if len({row.get("invocation_id") for row in rows}) != 1:
            valid.append(False)
        if len({row.get("request_sha256") for row in rows}) != 1:
            valid.append(False)

        completed = [row for row in rows if row.get("status") == "completed"]
        if len(completed) == 1:
            completed_fallbacks += 1
            final = completed[0]
            response = evidence.get("response")
            valid.extend([
                isinstance(response, dict),
                evidence.get("response_sha256") == signature(response),
                final.get("response_sha256") == evidence.get("response_sha256"),
                final.get("overflow_artifact_sha256") == file_hash(artifact_path),
            ])
        else:
            valid.append(False)
        if not all(valid):
            fallback_errors.append(artifact_name)

    unassociated_originals = sorted(
        set(original_overflow_attempts) - referenced_original_hashes)
    duplicate_original_references = (
        len(referenced_original_hashes) != len(fallback_groups))
    if unassociated_originals:
        fallback_errors.extend(unassociated_originals)
    if duplicate_original_references:
        fallback_errors.append("duplicate_original_overflow_reference")
    stats = {
        "attempts": len(records),
        "normal_truncate_false_attempts": sum(
            record.get("truncate") is False for record in records),
        "original_overflow_request_hashes": len(original_overflow_attempts),
        "original_overflow_invocations": len({
            row.get("invocation_id") for rows in original_overflow_attempts.values()
            for row in rows}),
        "original_overflow_attempts": sum(
            len(rows) for rows in original_overflow_attempts.values()
            if rows),
        "fallback_invocations": len(fallback_groups),
        "fallback_completed": completed_fallbacks,
        "fallback_retries": fallback_retry_count,
        "illegal_truncate_true": len(invalid_truncate),
        "invalid_identity": len(invalid_identity),
        "artifact_errors": len(artifact_errors),
        "fallback_errors": len(fallback_errors),
        "unassociated_original_overflows": len(unassociated_originals),
        "duplicate_original_references": int(duplicate_original_references),
        "authenticated_history_segments": len(history.segments) if history else 0,
        "history_errors": len(history_errors),
    }
    errors = {
        "invalid_identity": invalid_identity,
        "illegal_truncate_true": invalid_truncate,
        "artifact_errors": artifact_errors,
        "fallback_errors": fallback_errors,
        "history_errors": history_errors,
    }
    return stats, errors


def audit(campaign):
    findings = []

    def require(name, passed, evidence):
        findings.append({"requirement": name, "passed": bool(passed), "evidence": evidence})

    needed = [
        "status.json", "summary.json", "verification.json", "REPORT.md",
        "efficiency.json", "paper_efficiency.csv", "reproducibility.json",
        "environment.json", "embedding_contract.json", "provenance.json",
        "datasets.lock.json", "prompts.json", "execution.json",
        "resource_samples.jsonl", "phase_events.jsonl", "recovery_manifest.json",
        "independent_metrics.json", "api_calls.jsonl", "cache_hits.jsonl",
        "vector_index_audit.json",
    ]
    missing = [name for name in needed if not (campaign / name).is_file()]
    require("all_final_deliverables_exist", not missing, {"missing": missing})
    if missing:
        record = {"status": "incomplete", "checked_unix": time.time(), "findings": findings}
        write_json(campaign / "completion_audit.json", record)
        return record

    state, summary, verified = [read(campaign / name) for name in [
        "status.json", "summary.json", "verification.json"]]
    require("worker_completed_all_work", state.get("status") == "completed"
            and summary.get("status") == "completed", {
                "worker": state.get("status"), "summary": summary.get("status")})
    require("independent_full_verification", verified.get("status") == "verified_complete"
            and verified.get("full_benchmark") is True, {
                "status": verified.get("status"), "full_benchmark": verified.get("full_benchmark")})
    independent = read(campaign / "independent_metrics.json")
    require("dataset_primitive_metric_crosscheck_complete",
            independent.get("status") == "crosschecked_complete"
            and independent.get("full_benchmark") is True
            and independent.get("discrepancies") == [], {
                "status": independent.get("status"),
                "discrepancies": independent.get("discrepancies")})
    independent_rows = {row["dataset"]: row for row in independent["datasets"]}
    vector_audit = read(campaign / "vector_index_audit.json")
    vector_findings = {row["dataset"]: row for row in vector_audit["datasets"]}
    require("independent_all_vector_indexes_verified",
            vector_audit["status"] == "verified_vectors_complete"
            and vector_audit["full_three_indexes"] is True
            and vector_audit["failures"] == []
            and set(vector_findings) == {"hotpotqa", "2wiki", "musique"}, {
                "status": vector_audit["status"], "failures": vector_audit["failures"]})
    execution = read(campaign / "execution.json")
    require("full_3000_questions_and_32_workers",
            execution == {"limit": 1000, "datasets": ["hotpotqa", "2wiki", "musique"],
                          "workers": 32}, execution)
    provenance = read(campaign / "provenance.json")
    config = provenance["config"]
    require("correct_model_embedding_and_configuration", all([
        config["model"] == "openai/gpt-4o-mini",
        config["provider"] == "openrouter", config["temperature"] == 0,
        config["sample_seed"] == 20261020, config["embedding"] == "bge-m3:latest",
        config["embedding_dimension"] == 1024,
        config["chunk_token_size"] == 1200, config["chunk_overlap_token_size"] == 100,
        config["entity_extract_max_gleaning"] == 2,
        config["retrieval_mode"] == "hybrid", config["top_k"] == 60,
        config["inference_scope"] == "global_union_of_selected_candidate_passages",
        config["embedding_normalize"] is True,
        config["embedding_truncate"] is False,
        config["llm_seed"] is None,
        signature(config) == provenance["config_sha256"],
    ]), config)
    if config.get("embedding_overflow_policy"):
        policy = config["embedding_overflow_policy"]
        require("embedding_overflow_policy_is_explicit_and_narrow", all([
            policy.get("trigger") == "HTTP400_exact_input_length_exceeds_context_error",
            policy.get("primary_truncate") is False,
            policy.get("fallback_truncate") is True,
            policy.get("scope")
            == "all_embedding_inputs_without_gold_or_dataset_specific_selection",
            policy.get("in_range_cached_vectors")
            == "retain_prior_successful_truncate_false_vectors",
        ]), policy)
    refusal_policy = config.get("provider_refusal_policy", {})
    require("provider_filters_are_unavailable_and_never_retried", all([
        refusal_policy.get("explicit_content_filter_or_refusal")
        == "terminal_unavailable_output_no_retry",
        refusal_policy.get("query_extraction_or_answer")
        == "empty_answer_scored_in_full_frozen_denominator",
        refusal_policy.get("partial_filtered_text")
        == "provenance_only_never_used_as_evidence_or_answer",
    ]), refusal_policy)
    truncation_policy = config.get("output_truncation_policy", {})
    require("truncated_outputs_are_unavailable_under_unchanged_caps", all([
        truncation_policy.get("finish_reason_length") == "terminal_unavailable_output_no_retry",
        truncation_policy.get("output_token_caps") == "unchanged",
        truncation_policy.get("partial_truncated_text")
        == "provenance_only_never_used_as_evidence_or_answer",
        config.get("malformed_answer_policy")
        == "retain_raw_response_empty_answer_scored_full_denominator",
    ]), truncation_policy)
    incomplete_policy = config.get("output_incomplete_policy", {})
    require("missing_finish_outputs_are_separate_unavailable_outcomes", all([
        incomplete_policy.get("missing_or_null_finish_reason")
        == "terminal_unavailable_output_no_retry",
        incomplete_policy.get("cause") == "unknown_not_classified_as_filtering_or_token_cap",
        incomplete_policy.get("query_extraction_or_answer")
        == "empty_answer_scored_in_full_frozen_denominator",
        incomplete_policy.get("partial_text")
        == "provenance_only_never_used_as_evidence_or_answer",
    ]), incomplete_policy)
    source_errors = [
        name for name, expected in provenance["source_sha256"].items()
        if not (campaign / "source_snapshot" / name).is_file()
        or file_hash(campaign / "source_snapshot" / name) != expected]
    require("all_frozen_source_files_match", not source_errors, {"changed": source_errors})
    if config.get("embedding_runtime_scope") == "dedicated_local_Ollama_process_port11436":
        service = read(campaign / "embedding_service.json")
        contract = read(campaign / "embedding_contract.json")
        require("isolated_embedding_process_and_model_contract_recorded", all([
            service["endpoint"] == config["embedding_endpoint"] == contract["endpoint"]
            == "http://127.0.0.1:11436/api/embed",
            contract["digest"] == config["embedding_digest"],
            contract["execution_options"] == config["embedding_execution_options"],
            contract.get("overflow_policy") == config.get("embedding_overflow_policy"),
            contract["service_manifest_sha256"]
            == file_hash(campaign / "embedding_service.json"),
            service["scoped_environment"]["OLLAMA_HOST"] == "127.0.0.1:11436",
            bool(service.get("executable_sha256")),
        ]), {"endpoint": service["endpoint"], "service_pid_at_launch": service["pid"]})
    require("dataset_lock_is_unchanged", file_hash(campaign / "datasets.lock.json")
            == provenance["dataset_lock_sha256"], {"sha256": provenance["dataset_lock_sha256"]})
    environment = read(campaign / "environment.json")
    require("hardware_software_and_dependencies_recorded", all([
        bool(environment.get("python")), bool(environment.get("packages")),
        bool(environment.get("cpu_logical_count")),
        (campaign / "hardware_snapshots.jsonl").exists(),
        (campaign / "source_snapshot/requirements.lock.txt").exists(),
    ]), {"python": environment.get("python"), "package_count": len(environment.get("packages", {}))})

    lock = read(campaign / "datasets.lock.json")
    quality = {row["dataset"]: row for row in verified["datasets"]}
    efficiency = {row["dataset"]: row for row in read(campaign / "efficiency.json")["datasets"]}
    repro = read(campaign / "reproducibility.json")
    anchor_errors = [
        name for name, expected in repro["artifact_sha256"].items()
        if not (campaign / name).is_file()
        or file_hash(campaign / name) != expected]
    require("reproducibility_anchors_and_configuration_match",
            not anchor_errors and repro["config"] == config
            and repro["config_sha256"] == provenance["config_sha256"]
            and repro["source_sha256"] == provenance["source_sha256"], {
                "changed_anchors": anchor_errors})
    with (campaign / "paper_efficiency.csv").open(
            encoding="utf-8", newline="") as stream:
        paper_rows = list(csv.DictReader(stream))
    paper = {row["dataset"]: row for row in paper_rows}
    raw_calls = read_lines(campaign / "api_calls.jsonl")
    require("all_three_datasets_in_reports", set(quality) == set(efficiency) ==
            set(paper) == {"hotpotqa", "2wiki", "musique"}
            and len(paper_rows) == 3, {
                "metrics": sorted(quality), "efficiency": sorted(efficiency),
                "paper_table": sorted(paper)})
    if config.get("text_error_recovery"):
        policy = config["text_error_recovery"]
        require("explicit_provider_errors_use_bounded_exact_payload_retries", all([
            policy.get("max_attempts_per_exact_request_invocation") == 4,
            policy.get("model_prompt_cap_temperature_routing") == "unchanged_between_attempts",
            policy.get("partial_error_text") == "provenance_only_never_used_as_evidence_or_answer",
            policy.get("identity_checks_before_choice_error_retry") is True,
        ]), policy)
        groups = {}
        artifact_errors = []
        for row in raw_calls:
            if not row.get("request_invocation_id"):
                continue
            groups.setdefault(row["request_invocation_id"], []).append(row)
            artifact = row.get("failure_artifact")
            if artifact:
                original = read(campaign / artifact)
                if (signature(original["request"]) != row["request_sha256"]
                        or original.get("request_sha256") != row["request_sha256"]
                        or original.get("http_status") != row.get("http_status")):
                    artifact_errors.append(artifact)
                if row.get("provider_error"):
                    response = original.get("response") or {}
                    choices = response.get("choices") or [{}]
                    provider_error = response.get("error") or choices[0].get("error")
                    if (provider_error != row["provider_error"]
                            or row.get("usage_raw") != (response.get("usage") or {})):
                        artifact_errors.append(artifact)
        group_errors = [
            name for name, rows in groups.items()
            if len(rows) > 4 or len({row["request_sha256"] for row in rows}) != 1
            or [row["attempt"] for row in rows] != list(range(1, len(rows) + 1))
            or any(row.get("max_attempts") != 4 for row in rows)
            or any(row.get("retry_scheduled") is True and row["attempt"] >= 4 for row in rows)
            or any(row.get("retry_delay_s", 0) < 0 for row in rows)
        ]
        require("text_retry_groups_and_original_failure_artifacts_valid",
                not artifact_errors and not group_errors, {
                    "instrumented_invocations": len(groups),
                    "invalid_invocation_ids": group_errors,
                    "invalid_raw_artifacts": artifact_errors})
    embedding_history = None
    all_embedding_http = []
    if config.get("embedding_http_recovery"):
        try:
            embedding_history = load_authenticated_embedding_history(campaign, config)
            all_embedding_http = read_lines(campaign / "embedding_http_attempts.jsonl")
            history_stats, history_errors = audit_embedding_http_records(
                campaign, None, all_embedding_http, config, history=embedding_history)
            require("historical_embedding_contracts_authenticated",
                    not any(history_errors.values())
                    and all(row.get("dataset") in {"hotpotqa", "2wiki", "musique"}
                            for row in all_embedding_http),
                    {"statistics": history_stats, "errors": history_errors})
        except (EmbeddingHistoryError, OSError, KeyError, ValueError) as exc:
            require("historical_embedding_contracts_authenticated", False,
                    {"error": str(exc)})
    counters = Counter()
    for entry in lock["datasets"]:
        dataset = entry["dataset"]
        root = campaign / dataset
        source_ids = ROOT.parents[3] / entry["sources"]["ids"]["path"]
        frozen_ids = read(source_ids)
        manifest = read(root / "data_manifest.json")
        predictions = [read(path) for path in (root / "predictions").glob("*.json")]
        ids = [record["id"] for record in predictions]
        require(dataset + ":exact_frozen_1000_ids", all([
            len(ids) == len(set(ids)) == 1000, set(ids) == set(frozen_ids),
            manifest["ids"] == frozen_ids, repro["sample_ids"][dataset] == frozen_ids,
        ]), {"predictions": len(ids), "unique_ids": len(set(ids))})
        require(dataset + ":all_predictions_valid", all(
            record["status"] == "completed" and record["model"] == config["model"]
            and record["provider"] == "openrouter"
            and isinstance(record["answer"], str)
            and signature(record["context"]) == record["context_sha256"]
            for record in predictions), {"records_checked": len(predictions)})
        row, perf = quality[dataset], efficiency[dataset]
        calls = [record for record in raw_calls if record.get("dataset") == dataset]
        raw_totals = {
            "attempts": len(calls),
            "successful_calls": sum(record["status"] == "completed" for record in calls),
            "failed_attempts": sum(record["status"] != "completed" for record in calls),
            **{key: sum((record.get("usage") or {}).get(key, 0) or 0 for record in calls)
               for key in ("input_tokens", "output_tokens", "total_tokens")},
        }
        require(dataset + ":usage_matches_all_original_raw_attempts",
                all(perf["api"].get(key) == value for key, value in raw_totals.items()),
                raw_totals)
        unavailable = [record for record in calls
                       if record.get("status") in ("provider_filtered", "output_truncated", "output_incomplete")
                       or record.get("error") in (
                           "completion_content_filter", "completion_refusal", "completion_length",
                           "completion_None", "completion_missing_finish_reason")]
        invalid_terminal, retried_terminal = [], []
        for record in unavailable:
            key = record["request_sha256"]
            saved = read(campaign / "cache/responses" / (key + ".json"))
            original = saved.get("raw_response") or {}
            choices = original.get("choices") or [{}]
            terminal = choices[0]
            original_outcome_valid = (
                saved.get("status") == "provider_filtered"
                and (terminal.get("finish_reason") == "content_filter"
                     or (terminal.get("message") or {}).get("refusal"))
                or saved.get("status") == "output_truncated"
                and terminal.get("finish_reason") == "length"
                or saved.get("status") == "output_incomplete"
                and terminal.get("finish_reason") is None and len(choices) == 1
                and not (terminal.get("message") or {}).get("refusal"))
            if (saved.get("status") not in ("provider_filtered", "output_truncated", "output_incomplete")
                    or signature(saved["request"]) != key
                    or original.get("model") != saved.get("returned_model")
                    or terminal.get("finish_reason") != saved.get("finish_reason")
                    or not original_outcome_valid):
                invalid_terminal.append(key)
            if any(later["request_sha256"] == key
                   and later["started_unix"] > record["started_unix"]
                   for later in calls):
                retried_terminal.append(key)
        require(dataset + ":unavailable_raw_outputs_preserved_without_retry",
                not invalid_terminal and not retried_terminal, {
                    "unavailable_outcomes": len(unavailable),
                    "invalid_raw_outcomes": invalid_terminal,
                    "retried_terminal_requests": retried_terminal})
        independent_row = independent_rows[dataset]
        require(dataset + ":independent_metrics_and_predictions_match", all([
            independent_row["questions"] == 1000,
            all(math.isclose(independent_row[key], row[key], abs_tol=1e-10,
                             rel_tol=1e-12) for key in [
                "primary_em", "primary_f1", "native_alias_em", "native_alias_f1"]),
            independent_row["predictions_sha256"] == {
                path.name: file_hash(path) for path in (root / "predictions").glob("*.json")},
        ]), {"independently_scored_questions": independent_row["questions"]})
        filtered = [record for record in predictions
                    if record.get("answer_outcome") == "provider_filtered"]
        require(dataset + ":filtered_questions_remain_empty_and_counted", all(
            record["answer"] == "" and record.get("provider_refusal")
            for record in filtered)
            and row["provider_refusals"]["empty_answer_questions"] == len(filtered)
            and row["provider_refusals"] == perf["provider_refusals"],
            {"filtered_questions_in_full_denominator": len(filtered)})
        truncated = [record for record in predictions
                     if record.get("answer_outcome") == "output_truncated"]
        malformed = [record for record in predictions
                     if record.get("answer_outcome") == "invalid_answer_schema"]
        require(dataset + ":truncated_and_malformed_answers_remain_counted", all(
            record["answer"] == "" and record.get("unavailable_output")
            for record in truncated)
            and all(record["answer"] == "" for record in malformed)
            and row["output_truncations"]["empty_answer_questions"] == len(truncated)
            and row["output_truncations"] == perf["output_truncations"]
            and row["invalid_answer_schema_questions"]
            == perf["invalid_answer_schema_questions"] == len(malformed), {
                "truncated_questions": len(truncated),
                "malformed_answer_questions": len(malformed)})
        incomplete = [record for record in predictions
                      if record.get("answer_outcome") == "output_incomplete"]
        require(dataset + ":incomplete_answers_remain_empty_and_counted", all(
            record["answer"] == "" and record.get("unavailable_output")
            for record in incomplete)
            and row["incomplete_outputs"]["empty_answer_questions"] == len(incomplete)
            and row["incomplete_outputs"] == perf["incomplete_outputs"]
            and perf["api"]["incomplete_output_responses"] == sum(
                record.get("status") == "output_incomplete"
                or record.get("error") in ("completion_None", "completion_missing_finish_reason")
                for record in calls),
            {"incomplete_answer_questions": len(incomplete)})
        require(dataset + ":metrics_complete_and_finite", all([
            row["expected"] == row["completed"] == 1000,
            row["missing_or_failed"] == 0,
            all(math.isfinite(row[key]) and 0 <= row[key] <= 100 for key in [
                "primary_em", "primary_f1", "native_alias_em", "native_alias_f1"]),
            perf["primary_em"] == row["primary_em"], perf["primary_f1"] == row["primary_f1"],
        ]), {key: row[key] for key in ["expected", "completed", "primary_em", "primary_f1"]})
        require(dataset + ":timings_for_every_question", all(
            all(math.isfinite(record[key]) and record[key] >= 0 for key in [
                "latency_s", "retrieval_latency_s", "answer_generation_latency_s"])
            and record["latency_s"] + 0.02 >= (
                record["retrieval_latency_s"] + record["answer_generation_latency_s"])
            for record in predictions)
            and perf["query_latency"]["end_to_end"]["count"] == 1000
            and perf["wall_time"]["query_phase_wall_s"] > 0,
            {"count": perf["query_latency"]["end_to_end"]["count"]})
        attempts = perf.get("query_phase_attempts", [])
        require(dataset + ":query_wall_time_includes_interrupted_work",
                bool(attempts)
                and all(attempt.get("elapsed_s") is not None
                        and attempt["elapsed_s"] >= 0 for attempt in attempts)
                and math.isclose(
                    perf.get("query_all_attempts_wall_s", -1),
                    sum(attempt.get("elapsed_s") or 0 for attempt in attempts),
                    rel_tol=1e-12, abs_tol=1e-9)
                and math.isclose(
                    perf.get("query_all_attempts_questions_per_s") or -1,
                    1000 / perf["query_all_attempts_wall_s"],
                    rel_tol=1e-12, abs_tol=1e-9),
                {"attempts": attempts,
                 "all_attempts_wall_s": perf.get("query_all_attempts_wall_s")})
        require(dataset + ":full_corpus_and_valid_index", all([
            manifest["passage_count"] == entry["candidate_passage_count"],
            manifest["pilot"] is False,
            manifest["retrieval_filter"] == "none",
            row["graph"]["full_docs"] == entry["candidate_passage_count"],
            row["graph"]["chunks"] > 0, row["graph"]["entity_nodes"] > 0,
            row["graph"]["hyperedge_nodes"] > 0, row["graph"]["incidence_edges"] > 0,
            row["index_storage_bytes"] > 0,
        ]), row["graph"])
        vector_result = vector_findings.get(dataset, {})
        vector_files = vector_result.get("files", {})
        require(dataset + ":vector_files_still_match_independent_audit", len(vector_files) == 3
                and vector_result["index_pointer_sha256"]
                == file_hash(root / "index_complete.json")
                and all((campaign / name).is_file()
                        and file_hash(campaign / name) == info["file_sha256"]
                        and info["vectors_checked"] == info["expected_records"]
                        and info["dimensions"] == 1024
                        for name, info in vector_files.items()), {
                    "files_checked": len(vector_files)})
        require(dataset + ":construction_query_usage_and_storage", all([
            perf["construction_all_attempts_s"] > 0,
            perf["storage"]["persistent_index_bytes"] > 0,
            "construction" in perf["api_by_pipeline_phase"],
            "query" in perf["api_by_pipeline_phase"],
            perf["api"]["input_tokens"] > 0, perf["api"]["output_tokens"] > 0,
            perf["path_question_repository"]["applicable"] is False,
            perf["resource_observations"]["samples"] > 0,
        ]), {"llm_attempts": perf["api"]["attempts"],
             "index_bytes": perf["storage"]["persistent_index_bytes"],
             "construction_all_attempts_s": perf["construction_all_attempts_s"]})
        if config.get("embedding_http_recovery"):
            embedding_http = [
                record for record in all_embedding_http
                if record.get("dataset") == dataset]
            embedding_stats, embedding_errors = audit_embedding_http_records(
                campaign, dataset, embedding_http, config, history=embedding_history)
            require(dataset + ":embedding_http_attempts_bounded_and_traceable",
                    not any(embedding_errors.values())
                    and perf["embedding"]["http_attempts_since_instrumentation"]
                    == len(embedding_http)
                    and perf["embedding"]["http_failures_since_instrumentation"]
                    == sum(record["status"] != "completed"
                           for record in embedding_http),
                    {"attempts_since_instrumentation": len(embedding_http),
                     **embedding_stats, **embedding_errors})
        csv_values = {
            "questions": 1000, "em": row["primary_em"], "f1": row["primary_f1"],
            "construction_all_attempts_s": perf["construction_all_attempts_s"],
            "query_phase_wall_s": perf["wall_time"]["query_phase_wall_s"],
            "query_mean_s": perf["query_latency"]["end_to_end"]["mean_s"],
            "query_p50_s": perf["query_latency"]["end_to_end"]["p50_s"],
            "query_p95_s": perf["query_latency"]["end_to_end"]["p95_s"],
            "index_MiB": row["index_storage_bytes"] / 1024 ** 2,
            "input_tokens": perf["api"]["input_tokens"],
            "output_tokens": perf["api"]["output_tokens"],
            "workers": 32,
            "query_all_attempts_wall_s": perf["query_all_attempts_wall_s"],
            "query_all_attempts_questions_per_s": perf["query_all_attempts_questions_per_s"],
        }
        mismatches = [
            name for name, expected in csv_values.items()
            if not math.isclose(float(paper[dataset][name]), expected,
                                rel_tol=1e-9, abs_tol=1e-9)]
        require(dataset + ":paper_table_matches_verified_artifacts",
                not mismatches, {"mismatched_columns": mismatches})
        counters["predictions"] += len(predictions)

    recovery = read(campaign / "recovery_manifest.json")
    checkpoint_errors = [
        name for name, expected in recovery.get("imported_checkpoint_sha256", {}).items()
        if not (campaign / name).is_file() or file_hash(campaign / name) != expected]
    require("imported_completed_indexes_and_answers_unchanged",
            not checkpoint_errors, {
                "files_checked": len(recovery.get("imported_checkpoint_sha256", {})),
                "changed_files": checkpoint_errors})
    raw_artifact_errors = [
        name for name, expected in recovery.get("imported_raw_failure_artifact_sha256", {}).items()
        if not (campaign / name).is_file() or file_hash(campaign / name) != expected]
    require("original_raw_failure_artifacts_unchanged", not raw_artifact_errors, {
        "files_checked": len(recovery.get("imported_raw_failure_artifact_sha256", {})),
        "changed_files": raw_artifact_errors})
    history_errors = []
    for name, expected in recovery["inherited_history_sha256"].items():
        if not (campaign / "recovery_origin" / name).is_file() or file_hash(
                campaign / "recovery_origin" / name) != expected:
            history_errors.append(name)
    require("original_failed_work_and_cache_lineage_preserved", not history_errors
            and recovery["Luna_cache_imported"] is False
            and recovery["pilot_cache_imported"] is False, {
                "changed_history": history_errors,
                "imported_formal_responses": recovery["imported_response_count"]})
    prefix_errors = [
        name for name in recovery["inherited_history_sha256"]
        if not has_original_prefix(campaign / name, campaign / "recovery_origin" / name)]
    require("final_logs_include_every_original_history_record",
            not prefix_errors, {"modified_or_missing_history_prefix": prefix_errors})
    report = (campaign / "REPORT.md").read_text(encoding="utf-8")
    require("report_contains_quality_efficiency_and_limits", all(
        term in report.lower() for term in [
            "verified_complete", "efficiency measurements", "gpt-4o-mini",
            "path-question", "candidate", "provider-filtered outputs",
            "truncated and malformed outputs", "outputs without completion boundaries"]),
            {"path": str(campaign / "REPORT.md")})
    complete = all(row["passed"] for row in findings)
    record = {
        "status": "accepted_complete" if complete else "incomplete_or_failed_audit",
        "checked_unix": time.time(), "prediction_count": counters["predictions"],
        "audit_source_sha256": file_hash(Path(__file__)), "findings": findings,
    }
    write_json(campaign / "completion_audit.json", record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign", type=Path)
    args = parser.parse_args()
    result = audit(args.campaign)
    print(json.dumps(result, ensure_ascii=True, indent=2))
    raise SystemExit(0 if result["status"] == "accepted_complete" else 2)
