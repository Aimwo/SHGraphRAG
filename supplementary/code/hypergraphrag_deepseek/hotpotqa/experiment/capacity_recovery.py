"""Independent embedding-capacity recovery and immutable original history.

Copied into a prepared recovery's experiment package. No credentials or model
clients are opened by importing or auditing this module.
"""
from __future__ import annotations

from collections import Counter
from importlib import import_module, util
from pathlib import Path
import sys

from .common import ROOT, file_hash, read_json, read_jsonl, signature
from .text_cache_rekey import audit_entry, scopes


CAMPAIGN_NAME = "hypergraphrag_three_datasets_seed42_n1000_c32"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def require_model_calls_allowed():
    """Fresh public runs use explicit opt-in and a local stop marker."""
    import os
    if os.environ.get("SHGRAPHRAG_ALLOW_MODEL_API") != "1":
        raise RuntimeError("Model calls require explicit opt-in for this fresh run")
    if (ROOT / "STOP_MODEL_CALLS").exists():
        raise RuntimeError("STOP_MODEL_CALLS exists; remove it only to resume intentionally")


def source_root_for(manifest):
    source = (ROOT.parent / manifest["source_directory"]).resolve()
    require(source.parent == ROOT.parent.resolve() and source != ROOT.resolve(),
            "Original history escaped its preserved experiment directory")
    return source


def audit_source_identity(source, manifest, target_config):
    original = read_json(source / "experiment/config.json")
    scopes(original, target_config)
    require(original == manifest["original_config"], "Original configuration changed")
    require(read_json(source / "inputs/manifest.json")
            == read_json(ROOT / "inputs/manifest.json"), "Recovery input manifest differs")
    for name, expected in manifest["source_evidence_sha256"].items():
        require(file_hash(source / name) == expected,
                "Original pinned history changed: " + name)
    if manifest.get("original_provenance_sha256"):
        contract = read_json(source / "run/provenance.json")
        require(contract["config"] == original, "Original frozen configuration differs")
        for name, expected in contract["source_sha256"].items():
            require(file_hash(source / name)
                    == file_hash(source / "run/source_snapshot" / name) == expected,
                    "Original frozen source changed: " + name)
    else:
        require(not (source / "run/provenance.json").exists(),
                "An originally unstarted source has subsequently been launched")
    return original


def audit_prior_runtime_import(source, contract):
    """Run a prior recovery's own audit with its preserved original ROOT."""
    prior = contract.get("runtime_history_import")
    if not prior:
        return None
    alias = "_preserved_capacity_source_" + signature(str(source.resolve()))[:16]
    require(alias not in sys.modules, "Original audit package is already loaded")
    spec = util.spec_from_file_location(
        alias, source / "experiment/__init__.py",
        submodule_search_locations=[str(source / "experiment")])
    require(spec is not None and spec.loader is not None,
            "Original recovery audit package is unavailable")
    module = util.module_from_spec(spec)
    sys.modules[alias] = module
    try:
        spec.loader.exec_module(module)
        original_recovery = import_module(alias + ".recovery")
        result = original_recovery.audit_import(source / "run", prior)
        require(result is not None, "Original runtime recovery was not audited")
        return result
    finally:
        for name in list(sys.modules):
            if name == alias or name.startswith(alias + "."):
                del sys.modules[name]


def audit_physical_identity(row, artifact, kind, original_config, scope):
    request = artifact["request"]
    endpoint = (original_config["base_url"].rstrip("/") + "/chat/completions"
                if kind == "text" else
                original_config["embedding"]["base_url"] + "/api/embed")
    require(row["kind"] == kind
            and row["endpoint"] == artifact["endpoint"] == endpoint
            and row["scope_sha256"] == scope
            and row["request_sha256"] == signature({
                "scope": scope, "endpoint": endpoint, "body": request}),
            "Original physical request identity differs")
    require(row["active_requests_on_send"] <= 32
            and row["active_embedding_requests_on_send"] <= 4,
            "Original request exceeded its concurrency contract")
    if kind == "text":
        require(request["model"] == row["requested_model"]
                == original_config["model"]
                and row["credential_id"] == original_config["credential_id"],
                "Original text model or credential differs")
        if row.get("status") not in (
                None, "failed", "cancelled", "in_flight", "interrupted_outcome_unknown"):
            require(row.get("returned_model")
                    in original_config["allowed_returned_models"],
                    "Original returned model differs")
    else:
        require(request["model"] == original_config["embedding"]["model"]
                and request.get("options") == original_config["embedding"]["options"]
                and "truncate" not in request and len(request["input"]) <= 32,
                "Original embedding configuration differs")


def physical_source_history(source, manifest, original_config):
    """Audit true terminal attempts, and report observed interruption bounds."""
    run = source / "run"
    histories = {kind: read_jsonl(run / name) if (run / name).exists() else []
                 for kind, name in (
                     ("text", "api_calls.jsonl"), ("embedding", "embedding_calls.jsonl"))}
    starts = read_jsonl(run / "request_started.jsonl") if (
        run / "request_started.jsonl").exists() else []
    start_lookup = {row["attempt_id"]: row for row in starts}
    require(len(start_lookup) == len(starts), "Duplicate original physical start")
    scope, _ = scopes(original_config, manifest["target_config"])
    ends = {}
    for kind, ledger in histories.items():
        for row in ledger:
            attempt = row["attempt_id"]
            require(attempt not in ends, "Duplicate original terminal attempt")
            ends[attempt] = row
            require(attempt in start_lookup, "Original terminal request has no start")
            require(all(row.get(k) == value for k, value in start_lookup[attempt].items()
                        if k != "raw_usage"),
                    "Original start/terminal identity differs")
            artifact = read_json(run / row["raw_artifact"])
            require(artifact["record"] == row, "Original artifact and ledger differ")
            require(signature({k: v for k, v in artifact.items() if k != "record"})
                    == row["raw_payload_sha256"], "Original physical payload changed")
            audit_physical_identity(row, artifact, kind, original_config, scope)
    unknown_ids = set(start_lookup) - set(ends)
    interruption = manifest.get("user_interruption")
    require(unknown_ids == set((interruption or {}).get("attempt_ids", [])),
            "Original unclosed requests lack their exact interruption evidence")
    if interruption:
        proof = ROOT.parent / interruption["snapshot_relative_to_parent"]
        require(file_hash(proof) == interruption["snapshot_sha256"],
                "Original user-interruption snapshot changed")
        snapshot = read_json(proof)
        require({r["original_start_record"]["attempt_id"]
                 for r in snapshot["unknown_requests"]} == unknown_ids,
                "Original interruption snapshot names different requests")
        captured = {r["original_start_record"]["attempt_id"]: r
                    for r in snapshot["unknown_requests"]}
        for attempt in sorted(unknown_ids):
            row = start_lookup[attempt]
            require(captured[attempt]["original_start_record"] == row,
                    "Interrupted original start differs from the snapshot")
            artifact_path = run / row["raw_artifact"]
            require(file_hash(artifact_path) == captured[attempt]["source_artifact_sha256"],
                    "Interrupted original artifact changed")
            artifact = read_json(artifact_path)
            require(artifact["record"] == {**row, "status": "in_flight"}
                    and artifact.get("response_text") is None
                    and artifact.get("response_json") is None,
                    "Interrupted request has additional evidence requiring separate review")
            audit_physical_identity(row, artifact, row["kind"], original_config, scope)
            require(row["started_unix"] <= interruption["client_stop_observed_unix"],
                    "Interrupted request starts after the observed client stop")
            # A reporting record, never written into the original physical ledger.
            histories[row["kind"]].append({
                **row, "status": "interrupted_outcome_unknown",
                "finished_unix": interruption["client_stop_observed_unix"],
                "finished_unix_is_client_stop_upper_bound": True,
                "server_finish_time_known": False, "raw_usage": None,
                "raw_response_observed": False,
            })
    return histories


def original_times(source, manifest):
    run = source / "run"
    construction = read_jsonl(run / "construction_attempts.jsonl") if (
        run / "construction_attempts.jsonl").exists() else []
    invocations = read_jsonl(run / "campaign_attempts.jsonl") if (
        run / "campaign_attempts.jsonl").exists() else []
    phases = read_jsonl(run / "phase_events.jsonl") if (
        run / "phase_events.jsonl").exists() else []
    result = {
        "completed_construction_attempt_wall_s": sum(r["wall_s"] for r in construction),
        "completed_invocation_wall_s": sum(r["wall_s"] for r in invocations),
        "observed_interrupted_construction_wall_s_upper_bound": 0,
        "observed_interrupted_invocation_wall_s_upper_bound": 0,
    }
    interruption = manifest.get("user_interruption")
    if interruption:
        stop = interruption["client_stop_observed_unix"]
        state = read_json(run / "status.json")
        invocation_start = state["started_unix"]
        require(not any(r.get("started_unix") == invocation_start for r in invocations),
                "Interrupted invocation already has a terminal timing record")
        result["observed_interrupted_invocation_wall_s_upper_bound"] = stop - invocation_start
        latest_construction = max(
            (r["time_unix"] for r in phases if r["phase"] == "construction_started"
             and r["time_unix"] >= invocation_start), default=None)
        if latest_construction is not None:
            require(not any(r.get("started_unix", 0) >= invocation_start
                            for r in construction),
                    "Interrupted construction already has a terminal timing record")
            result["observed_interrupted_construction_wall_s_upper_bound"] = (
                stop - latest_construction)
    result["construction_wall_s_including_observed_upper_bounds"] = (
        result["completed_construction_attempt_wall_s"]
        + result["observed_interrupted_construction_wall_s_upper_bound"])
    result["invocation_wall_s_including_observed_upper_bounds"] = (
        result["completed_invocation_wall_s"]
        + result["observed_interrupted_invocation_wall_s_upper_bound"])
    return result


def audit_capacity_import(run, manifest, target_config):
    if manifest and manifest.get("recovery_kind") == "summary_visible_limit_repair":
        from .summary_reuse import audit_summary_reuse
        return audit_summary_reuse(run, manifest, target_config)
    if not manifest:
        return {"text": [], "embedding": []}, None
    require(manifest == read_json(ROOT / "provenance/capacity_recovery.json"),
            "Frozen capacity recovery manifest changed")
    require(manifest["target_config"] == target_config, "Target configuration differs")
    source = source_root_for(manifest)
    original = audit_source_identity(source, manifest, target_config)
    contract_file = source / "run/provenance.json"
    prior_audit = audit_prior_runtime_import(
        source, read_json(contract_file) if contract_file.exists() else {})
    histories = physical_source_history(source, manifest, original)
    mapping_file = ROOT / manifest["cache_mapping"]
    require(file_hash(mapping_file) == manifest["cache_mapping_sha256"],
            "Cache migration mapping changed")
    mappings = read_jsonl(mapping_file)
    require(len(mappings) == manifest["imported_terminal_text_caches"],
            "Imported cache mapping count differs")
    seen = set()
    for mapping in mappings:
        key = mapping["proposed_cache_key"]
        require(key not in seen, "Duplicate imported text cache key")
        seen.add(key)
        target = read_json(run / "cache/responses" / (key + ".json"))
        audit_entry(
            target,
            source_cache_bytes=(source / "run" / mapping["source_cache"]).read_bytes(),
            source_artifact_bytes=(source / "run" / mapping["source_artifact"]).read_bytes(),
            mapping=mapping, source_config=original, target_config=target_config)
    times = original_times(source, manifest)
    summary = {
        "source_directory": source.name,
        "prior_runtime_history_import_audit": prior_audit,
        "original_physical_attempts": {k: len(v) for k, v in histories.items()},
        "original_attempt_outcomes": {
            k: dict(Counter(r["status"] for r in v)) for k, v in histories.items()},
        "imported_terminal_text_caches": len(mappings),
        "imported_vectors": 0,
        "unknown_interrupted_outcomes": sum(
            r["status"] == "interrupted_outcome_unknown"
            for records in histories.values() for r in records),
        "timing": times,
        "history_scope_note": (
            "Original physical identities, input/response evidence and usage are "
            "audited under the original embedding scope. Cache lookup keys alone "
            "use the new scope. Unknown interrupted outcomes and usage remain "
            "unknown; client stop observations bound their client intervals."),
    }
    return histories, summary
