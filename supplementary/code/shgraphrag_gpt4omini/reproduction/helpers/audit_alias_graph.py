"""Verify full alias decisions and graph from saved inputs/responses, without API/gold."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from reproduction import prompts
from reproduction.client import RUNTIME_DIRECTORY, canonical, parse_json, request_body, signature, write_json
from reproduction.data import read_jsonl, sha256
from reproduction.entity_resolution import (
    candidate_pairs, canonical_mapping, catalog, remap_extractions)
from reproduction.graph import assemble, path_statistics
from reproduction.helpers.audit_response_artifacts import response_for


def audit(run):
    started = time.perf_counter()
    for name, digest in json.loads((run / "code_manifest.json").read_text(
            encoding="utf-8"))["modules_sha256"].items():
        if sha256(ROOT / "reproduction" / name) != digest:
            raise ValueError("Research code drift")
    corpus = read_jsonl(run / "inputs/corpus.jsonl")
    groups = json.loads((run / "inputs/construction_groups.json").read_text(encoding="utf-8"))
    extraction = {
        p["passage_id"]: json.loads((run / "extractions" / (p["passage_id"] + ".json"))
                                    .read_text(encoding="utf-8"))["extracted"]
        for p in corpus}
    details, frequencies = catalog(extraction)
    resolution = run / "entity_resolution"
    manifest = json.loads((resolution / "manifest.json").read_text(encoding="utf-8"))
    candidates = json.loads((resolution / "candidates.json").read_text(encoding="utf-8"))
    fingerprint = signature({
        "extracted": extraction, "groups": groups,
        "model_digest": manifest["embedding_model_digest"],
        "threshold": 0.95, "prompt": prompts.MERGE_SYSTEM})
    if fingerprint != manifest["input_fingerprint"] or fingerprint != candidates["input_fingerprint"]:
        raise ValueError("Alias inputs or fingerprint drift")
    database = ROOT / RUNTIME_DIRECTORY / "embedding_cache.sqlite3"
    vectors = []
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
        for entity in details.values():
            key = signature({"model_digest": manifest["embedding_model_digest"],
                             "text": entity["name"]})
            row = connection.execute("SELECT vector FROM embeddings WHERE key=?", (key,)).fetchone()
            if row is None:
                raise ValueError("Missing source alias-candidate vector")
            vectors.append(np.frombuffer(row[0], dtype=np.float32))
    vectors = np.stack(vectors)
    rebuilt_candidates = list(candidate_pairs(groups, extraction, details, vectors))
    if rebuilt_candidates != candidates["pairs"]:
        raise ValueError("Candidate rule/coverage differs from exact saved BGE vectors")
    del vectors
    decisions, proofs = [], []
    expected_files = set()
    for start in range(0, len(rebuilt_candidates), 10):
        batch = rebuilt_candidates[start:start + 10]
        payload = [{"id": f"p{i}", "entities": [
            {k: details[key][k] for k in ("name", "type", "description")}
            for key in pair["keys"]]} for i, pair in enumerate(batch)]
        expected = signature([prompts.MERGE_SYSTEM, payload])
        expected_files.add(expected + ".json")
        artifact_path = resolution / "batches" / (expected + ".json")
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        ids = {p["id"] for p in payload}
        schema = prompts.obj({"pairs": {
            "type": "array", "minItems": len(batch), "maxItems": len(batch),
            "items": prompts.obj({"id": {"type": "string", "enum": sorted(ids)},
                                  "same_entity": prompts.BOOL})}})
        original = canonical({"pairs": payload})
        prompt = original
        response_proofs = []
        for repair in range(3):
            response, proof = response_for(request_body(
                prompts.MERGE_SYSTEM, prompt, 2048, schema))
            response_proofs.append(proof)
            try:
                value = parse_json(response["text"])
                rows = value["pairs"]
                if (len(rows) != len(ids) or {r["id"] for r in rows} != ids
                        or any(not isinstance(r["same_entity"], bool) for r in rows)):
                    raise ValueError("Invalid entity-pair confirmation coverage")
                break
            except (ValueError, TypeError, KeyError, AssertionError) as error:
                if repair == 2:
                    raise ValueError("Saved alias correction remained invalid") from error
                prompt = (original + "\nThe previous output did not match "
                          "the required complete JSON schema. Return a full corrected "
                          "replacement, with no extra text. Error: " + str(error)
                          + "\nPrevious output:\n" + response["text"])
        saved_rows = [{"keys": batch[int(row["id"][1:])]["keys"],
                       "same_entity": row["same_entity"]} for row in value["pairs"]]
        if (artifact["input_fingerprint"] != expected
                or artifact["request_sha256"] != response["request_sha256"]
                or artifact["decisions"] != saved_rows):
            raise ValueError("Saved alias decisions differ from accepted response")
        decisions.extend(saved_rows)
        proofs.append({"batch": expected, "artifact_sha256": sha256(artifact_path),
                       "responses": response_proofs})
    if {p.name for p in (resolution / "batches").glob("*.json")} != expected_files:
        raise ValueError("Unexpected or missing alias batches")
    mapping = canonical_mapping(details, frequencies, sorted(
        tuple(row["keys"]) for row in decisions if row["same_entity"]))
    if mapping != json.loads((resolution / "mapping.json").read_text(encoding="utf-8")):
        raise ValueError("Alias mapping inconsistent with response decisions")
    expected_graph = assemble(corpus, remap_extractions(extraction, details, mapping))
    graph = json.loads((run / "graph.json").read_text(encoding="utf-8"))
    for field in ("entities", "hyperedges", "full_triples", "triples",
                  "bridge_ids", "retained_entity_ids"):
        if expected_graph[field] != graph[field]:
            raise ValueError("Graph does not match source extractions and confirmed aliases")
    stats = path_statistics(graph)
    for key, value in stats.items():
        if graph["stats"][key] != value:
            raise ValueError("Graph path statistics differ")
    return {
        "status": "passed", "observed_unix": time.time(),
        "elapsed_verification_s": time.perf_counter() - started,
        "scope": "complete_alias_and_graph_component_provenance",
        "new_model_calls": 0, "gold_read": False, "source_vectors_regenerated": False,
        "candidate_pairs": len(rebuilt_candidates), "confirmed_batches": len(proofs),
        "full_graph_sha256": sha256(run / "graph.json"),
        "two_edge_paths": stats["two_edge_paths"], "checks": [
            "frozen_code", "exact_BGE_candidate_coverage", "complete_request_reconstruction",
            "saved_alias_decisions_equal_accepted_responses", "canonical_mapping",
            "full_and_sparse_graph_rebuilt", "path_count_formula"],
        "batches": proofs,
        "limitation": "Artifact/policy provenance, not semantic correctness or QA accuracy. "
                      "Full path repository, indexes and predictions require final verification.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    from reproduction.helpers.storage_paths import checked_run_path
    run = checked_run_path(args.run)
    result = audit(run)
    write_json(run / "alias_graph_provenance_audit.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "batches"}, indent=2))
