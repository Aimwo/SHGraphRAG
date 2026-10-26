"""Validate reversible metadata repair and exact same-campaign cache imports."""
from __future__ import annotations

import json
from pathlib import Path
import re
import sqlite3

from .common import ROOT, file_hash, read_json, read_jsonl, signature

MARKER = "[REDACTED_KEY]"


def decode_empty_key_redaction(value):
    """Undo only the proven single-character interleaving bug.

    Literal ordinary redaction markers, provider responses and missing values
    are never inferred or filled in. Original artifacts remain preserved.
    """
    if isinstance(value, dict):
        return {key: decode_empty_key_redaction(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode_empty_key_redaction(item) for item in value]
    if isinstance(value, str) and value.startswith(MARKER) and value.endswith(MARKER):
        parts = value.split(MARKER)
        if parts[0] == parts[-1] == "" and all(len(item) == 1 for item in parts[1:-1]):
            return "".join(parts[1:-1])
    return value


def audit_import(run, manifest):
    if not manifest:
        return None
    if manifest != read_json(ROOT / "provenance/runtime_recovery.json"):
        raise ValueError("Frozen runtime recovery manifest changed")
    for relative, expected in manifest["immutable_imported_files"].items():
        if file_hash(run / relative) != expected:
            raise ValueError("Imported evidence changed: " + relative)
    original_contract = read_json(run / "imported_history/original_provenance.json")
    if (original_contract["config"] != read_json(ROOT / "experiment/config.json")
            or original_contract["inputs"] != read_json(ROOT / "inputs/manifest.json")):
        raise ValueError("Imported campaign configuration or frozen inputs differ")
    for relative, expected in original_contract["source_sha256"].items():
        if file_hash(run / "imported_history/source_snapshot" / relative) != expected:
            raise ValueError("Original frozen source snapshot changed")
    for relative, information in manifest["append_only_imported_prefixes"].items():
        import hashlib
        with (run / relative).open("rb") as stream:
            prefix = stream.read(information["bytes"])
        if hashlib.sha256(prefix).hexdigest() != information["sha256"]:
            raise ValueError("Imported history prefix changed: " + relative)
    original_starts = read_jsonl(run / "imported_history/original_request_started.jsonl")
    normalized_starts = read_jsonl(run / "request_started.jsonl")[:len(original_starts)]
    if [decode_empty_key_redaction(row) for row in original_starts] != normalized_starts:
        raise ValueError("Imported request starts differ from their proven transformation")
    migration = read_jsonl(run / "imported_history/request_migration.jsonl")
    chunks = None
    for entry in migration:
        original = read_json(run / entry["preserved_original"])
        current = read_json(run / entry["normalized_artifact"])
        original_row = original["record"]
        decoded = decode_empty_key_redaction(original_row)
        if signature({key: value for key, value in original.items() if key != "record"}) != decoded["raw_payload_sha256"]:
            raise ValueError("Original raw artifact does not match its original recorded hash")
        repaired = decode_empty_key_redaction(original)
        if entry["lossless_metadata_repair"]:
            if original.get("response_text") is not None or decoded["status"] not in ("failed", "cancelled"):
                raise ValueError("Repair scope expanded beyond empty-response exit metadata")
        restored = entry.get("restored_candidate_input_positions", {})
        if restored:
            if chunks is None:
                from hypergraphrag.operate import chunking_by_token_size
                import hashlib
                chunks = {}
                for passage in read_jsonl(ROOT / "inputs/candidate_corpus.jsonl"):
                    for chunk in chunking_by_token_size(
                            passage["text"].strip(), 100, 1200, tiktoken_model="gpt-4o-mini"):
                        text = chunk["content"]
                        chunks[hashlib.sha256(text.encode("utf-8")).hexdigest()] = text
            if repaired["record"]["kind"] != "embedding":
                raise ValueError("Candidate-input reconstruction scope expanded")
            for index, content_hash in restored.items():
                text = chunks[content_hash]
                position = int(index)
                if re.sub(r"sk-[A-Za-z0-9_-]+", MARKER, text) != repaired["request"]["input"][position]:
                    raise ValueError("Restored candidate input differs from its original recorded redaction")
                repaired["request"]["input"][position] = text
            if signature({
                    "scope": repaired["record"]["scope_sha256"],
                    "endpoint": repaired["endpoint"], "body": repaired["request"]
                    }) != repaired["record"]["request_sha256"]:
                raise ValueError("Reconstructed public input does not match the original sent-request hash")
        repaired["record"]["raw_payload_sha256"] = signature({
            key: value for key, value in repaired.items() if key != "record"})
        if repaired != current:
            raise ValueError("Imported attempt differs from the proven lossless transformation")
    expected_vectors = read_json(run / "imported_history/vector_hashes.json")
    import hashlib
    with sqlite3.connect((run / "cache/embeddings.sqlite").as_uri() + "?mode=ro", uri=True) as db:
        for key, expected in expected_vectors.items():
            value = db.execute("SELECT vector FROM vectors WHERE key=?", (key,)).fetchone()
            if value is None or hashlib.sha256(value[0]).hexdigest() != expected:
                raise ValueError("An imported same-campaign vector changed")
    return {"verified_imported_attempts": len(migration),
            "losslessly_repaired_exit_metadata": sum(entry["lossless_metadata_repair"] for entry in migration),
            "restored_public_candidate_inputs": sum(
                len(entry.get("restored_candidate_input_positions", {})) for entry in migration),
            "verified_imported_vectors": len(expected_vectors),
            "responses_invented": 0}
