"""Replay global path/entity selections from retained vectors, without inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from reproduction.client import signature, write_json
from reproduction.data import sha256
from reproduction.embedding import VectorIndex
from reproduction.helpers.audit_qa_response_provenance import QAAudit
from reproduction.helpers.audit_response_artifacts import load, require
from reproduction.helpers.storage_paths import checked_run_path


def compare(actual, expected, fields, label):
    require([{k: h[k] for k in fields} for h in actual]
            == [{k: h[k] for k in fields} for h in expected], label + " ranking differs")
    differences = [abs(a["score"] - e["score"]) for a, e in zip(actual, expected)]
    require(all(value <= 1e-6 for value in differences), label + " scores differ")
    return max(differences, default=0.0)


def audit(run):
    began = time.perf_counter()
    core = load(run / "code_manifest.json")["modules_sha256"]
    require(core == {p.name: sha256(p) for p in (ROOT / "reproduction").glob("*.py")},
            "Frozen research code differs")
    questions = load(run / "inputs/questions.json")
    require(all(set(q) == {"id", "question"} for q in questions), "Unexpected input labels")
    require([q["id"] for q in questions] == load(run / "inputs/ids.json"),
            "Frozen question order differs")
    manifest = load(run / "data_manifest.json")
    for name, expected in manifest["inputs_sha256"].items():
        require(sha256(run / "inputs" / name) == expected, "Frozen inference inputs differ")
    api_hash = sha256(run / "api_calls.jsonl")
    saved_index_audit = load(run / "index_provenance_audit.json")
    require(saved_index_audit["status"] == "passed", "Complete indexes not independently accepted")
    indexes = {}
    for name in ("paths", "entities"):
        accepted = next(r for r in saved_index_audit["indexes"] if r["name"] == name)
        metadata_path = run / "indexes" / (name + ".json")
        matrix_path = run / "indexes" / (name + ".npy")
        require(sha256(metadata_path) == accepted["metadata_sha256"]
                and sha256(matrix_path) == accepted["matrix_sha256"],
                "Accepted index bytes differ")
        indexes[name] = VectorIndex(
            load(metadata_path)["items"], np.load(matrix_path, mmap_mode="r"))
    qa = QAAudit(run)
    results, max_score_error, entity_queries = [], 0.0, 0
    try:
        require(set(qa.predictions) == {q["id"] for q in questions}, "Prediction coverage differs")
        for index, question in enumerate(questions):
            prediction = qa.predictions[question["id"]]
            require(prediction["question"] == question["question"], "Question text differs")
            raw_hits = indexes["paths"].search(qa.vector(question["question"]), 6)
            selected, seen = [], set()
            for hit in raw_hits:
                if hit["path_id"] not in seen:
                    selected.append(hit)
                    seen.add(hit["path_id"])
                    if len(selected) == 3:
                        break
            max_score_error = max(max_score_error, compare(
                selected, prediction["trace"][0]["path_hits"],
                ("id", "path_id"), "Global direct-path"))
            entity_evidence = []
            for step in prediction["trace"][2:]:
                require(step["stage"] == "subquestion", "Unknown retained retrieval step")
                by_id = {}
                for name in step["rewritten"]["entities"]:
                    entity_queries += 1
                    for hit in indexes["entities"].search(qa.vector(name), 2):
                        if hit["score"] > by_id.get(hit["id"], {}).get("score", -2):
                            by_id[hit["id"]] = hit
                linked = sorted(by_id.values(), key=lambda h: (-h["score"], h["id"]))[:2]
                linked = [hit for hit in linked if hit["score"] >= 0.3]
                max_score_error = max(max_score_error, compare(
                    linked, step["entity_hits"], ("id",), "Global entity"))
                entity_evidence.append(signature(linked))
            results.append({
                "id": question["id"],
                "prediction_sha256": sha256(run / "predictions" / (question["id"] + ".json")),
                "global_top_six_path_questions_sha256": signature(raw_hits),
                "selected_distinct_paths_sha256": signature(selected),
                "entity_selections_sha256": entity_evidence,
            })
            if (index + 1) % 100 == 0:
                print(json.dumps({
                    "retrieval_questions_replayed": index + 1, "expected": len(questions)}),
                      flush=True)
        require(sha256(run / "api_calls.jsonl") == api_hash, "Remote log changed during replay")
        result = {
            "status": "passed", "observed_unix": time.time(),
            "elapsed_s": time.perf_counter() - began,
            "questions": len(results),
            "complete_prediction_coverage": len(results) == manifest["questions"],
            "global_path_searches_replayed": len(results),
            "global_entity_searches_replayed": entity_queries,
            "maximum_saved_score_absolute_error": max_score_error,
            "score_absolute_tolerance": 1e-6,
            "new_model_calls": 0, "new_embedding_calls": 0, "gold_read": False,
            "helper_sha256": sha256(Path(__file__)), "core_hashes": core,
            "index_provenance_audit_sha256": sha256(run / "index_provenance_audit.json"),
            "remote_api_log_sha256_unchanged": api_hash,
            "retained_query_vectors_verified": qa.vector_proofs,
            "question_evidence": results,
            "scope": "Exact global cosine path/entity searches and stable ID/order "
                     "replayed with retained normalized query vectors. Hyperedge "
                     "evidence selection is checked separately by the complete QA audit.",
            "limitations": [
                "Uses retained BGE query vectors; does not regenerate them.",
                "Replay wall time is separate from original measured query time.",
            ],
        }
    finally:
        qa.connection.close()
    write_json(run / "retrieval_replay_audit.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    result = audit(checked_run_path(args.run))
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("question_evidence", "retained_query_vectors_verified",
                                   "core_hashes")}, indent=2))
