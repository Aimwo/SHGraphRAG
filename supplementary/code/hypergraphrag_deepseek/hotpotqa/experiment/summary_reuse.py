"""Audit a same-provider summary repair and all its preserved physical history."""
from __future__ import annotations

from importlib import import_module, util
import sqlite3
import sys

from .common import ROOT, file_hash, read_json, read_jsonl, signature


ORIGIN = "_history_source_directory"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def original_capacity_audit(source, contract, config):
    alias = "_summary_repair_source_" + signature(str(source))[:16]
    spec = util.spec_from_file_location(
        alias, source / "experiment/__init__.py",
        submodule_search_locations=[str(source / "experiment")])
    require(spec is not None and spec.loader is not None, "Original audit package missing")
    package = util.module_from_spec(spec)
    sys.modules[alias] = package
    try:
        spec.loader.exec_module(package)
        module = import_module(alias + ".capacity_recovery")
        return module.audit_capacity_import(
            source / "run", contract["capacity_recovery_import"], config)
    finally:
        for name in list(sys.modules):
            if name == alias or name.startswith(alias + "."):
                del sys.modules[name]


def audit_seeded_validation(run):
    manifest_path = ROOT / "provenance/summary_validation_import.json"
    if not manifest_path.exists():
        return None
    manifest = read_json(manifest_path)
    source = (ROOT.parent / manifest["source_relative_to_day"]).resolve()
    require(source.is_relative_to(ROOT.parent), "Validation source escaped this experiment day")
    require(file_hash(source / "result.json") == manifest["source_result_sha256"],
            "Original validation result changed")
    for relative, expected in manifest["source_files_sha256"].items():
        require(file_hash(source / relative) == expected, "Original validation evidence changed")
        if relative in ("run/api_calls.jsonl", "run/request_started.jsonl"):
            original_rows = read_jsonl(source / relative)
            target_rows = {row["attempt_id"]: row for row in read_jsonl(ROOT / relative)}
            require(all(target_rows.get(row["attempt_id"]) == row for row in original_rows),
                    "A preloaded validation record differs from its real physical identity")
        else:
            require(file_hash(ROOT / relative) == expected,
                    "A preloaded validation artifact or response cache changed")
    return {
        "physical_attempts": len(manifest["preloaded_physical_attempt_ids"]),
        "physical_attempt_ids": manifest["preloaded_physical_attempt_ids"],
        "source_relative_to_day": manifest["source_relative_to_day"],
        "manifest_sha256": file_hash(manifest_path),
        "counted_as_preserved_physical_attempts_not_new_requests": True,
    }


def audit_summary_reuse(run, manifest, target_config):
    source = (ROOT.parent / manifest["source_directory"]).resolve()
    require(source.parent == ROOT.parent and source != ROOT, "Invalid preserved repair source")
    require(manifest == read_json(ROOT / "provenance/capacity_recovery.json"),
            "Summary repair manifest changed")
    original_config = read_json(source / "experiment/config.json")
    require(original_config == target_config == manifest["original_config"]
            == manifest["target_config"], "Summary repair changes the backend configuration")
    for relative, expected in manifest["source_evidence_sha256"].items():
        require(file_hash(source / relative) == expected,
                "Preserved repair evidence changed: " + relative)
    contract = read_json(source / "run/provenance.json")
    require(contract["config"] == original_config, "Original frozen backend changed")
    for relative, expected in contract["source_sha256"].items():
        require(file_hash(source / relative) == expected
                == file_hash(source / "run/source_snapshot" / relative),
                "Original frozen source changed: " + relative)
    from .verify import audit_requests
    current = audit_requests(source / "run", original_config)
    prior, prior_audit = original_capacity_audit(source, contract, original_config)
    history = {}
    prior_name = prior_audit["source_directory"]
    for kind in ("text", "embedding"):
        history[kind] = [
            {**row, ORIGIN: row.get(ORIGIN, prior_name)} for row in prior[kind]
        ] + [{**row, ORIGIN: source.name} for row in current[kind]]
    ids = [row["attempt_id"] for values in history.values() for row in values]
    require(len(ids) == len(set(ids)), "Summary reuse counts a physical request twice")
    mapping_path = ROOT / manifest["cache_mapping"]
    require(file_hash(mapping_path) == manifest["cache_mapping_sha256"],
            "Summary cache mapping changed")
    mappings = read_jsonl(mapping_path)
    require(len(mappings) == manifest["imported_terminal_text_caches"],
            "Copied terminal cache count differs")
    for row in mappings:
        original = source / "run" / row["source_cache"]
        target = run / "cache/responses" / (row["proposed_cache_key"] + ".json")
        require(file_hash(original) == file_hash(target) == row["source_cache_sha256"],
                "An exact preserved response changed during summary repair")
    vector_info = manifest["embedding_cache"]
    database = source / vector_info["source_file"]
    require(file_hash(database) == vector_info["source_file_sha256"],
            "Preserved local embedding cache changed")
    with sqlite3.connect(run / "cache/embeddings.sqlite") as connection:
        connection.execute("ATTACH DATABASE ? AS preserved", (str(database),))
        count = connection.execute("SELECT count(*) FROM preserved.vectors").fetchone()[0]
        mismatch = connection.execute(
            "SELECT count(*) FROM preserved.vectors s LEFT JOIN main.vectors t "
            "ON t.key=s.key WHERE t.key IS NULL OR t.vector!=s.vector"
        ).fetchone()[0]
    require(count == vector_info["vectors"] and mismatch == 0,
            "The exact source embedding vectors were not retained")
    from .capacity_recovery import original_times
    times = original_times(source, manifest)
    prior_times = prior_audit["timing"]
    timing = {
        key: times.get(key, 0) + prior_times.get(key, 0)
        for key in set(times) | set(prior_times)
        if isinstance(times.get(key, prior_times.get(key)), (int, float))
    }
    source_manifest = contract["capacity_recovery_import"]
    sources = prior_audit.get("history_sources") or [{
        "source_directory": prior_name,
        "source_evidence_sha256": source_manifest["source_evidence_sha256"],
        "user_interruption": source_manifest.get("user_interruption"),
        "original_physical_attempts": prior_audit["original_physical_attempts"],
    }]
    sources = [*sources, {
        "source_directory": source.name,
        "source_evidence_sha256": manifest["source_evidence_sha256"],
        "user_interruption": None,
        "original_physical_attempts": {kind: len(rows) for kind, rows in current.items()},
    }]
    return history, {
        "source_directory": source.name,
        "recovery_kind": manifest["recovery_kind"],
        "history_sources": sources,
        "prior_capacity_import_audit": prior_audit,
        "original_physical_attempts": {kind: len(rows) for kind, rows in history.items()},
        "imported_terminal_text_caches": len(mappings),
        "imported_vectors": count, "timing": timing,
        "unknown_interrupted_outcomes": sum(
            row["status"] == "interrupted_outcome_unknown"
            for values in history.values() for row in values),
        "history_scope_note": (
            "Original c32 and c100 attempts retain their distinct actual inputs, "
            "responses, timing and usage. Exact same-input BGE vectors are reused; "
            "changed summary descriptions generate new embeddings."
        ),
    }
