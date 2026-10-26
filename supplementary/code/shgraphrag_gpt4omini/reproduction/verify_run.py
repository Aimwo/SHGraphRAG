"""Independent artifact checks before any completion claim."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .client import EFFORT, MODEL, PROVIDER, signature, write_json
from .data import read_jsonl, sha256
from .preflight import check_original


def verify(root: Path) -> dict:
    preserved = check_original()
    if preserved["source_files_changed"] or not preserved["working_tree_status_unchanged"]:
        raise RuntimeError("Original tree changed relative to recorded snapshot")
    data = json.loads((root / "data_manifest.json").read_text(encoding="utf-8"))
    for item in data["source_files"].values():
        if sha256(Path(item["path"])) != item["sha256"]:
            raise RuntimeError("Source dataset hash drift")
    for name, expected in data["inputs_sha256"].items():
        if sha256(root / "inputs" / name) != expected:
            raise RuntimeError("Inference input hash drift")
    questions = json.loads((root / "inputs/questions.json").read_text(encoding="utf-8"))
    ids = json.loads((root / "inputs/ids.json").read_text(encoding="utf-8"))
    if [r["id"] for r in questions] != ids or any(set(q) != {"id", "question"} for q in questions):
        raise RuntimeError("Question alignment or annotation boundary failure")
    corpus = read_jsonl(root / "inputs/corpus.jsonl")
    source_ids = {p["passage_id"] for p in corpus}
    extracted = {p.stem for p in (root / "extractions").glob("*.json")}
    if extracted != source_ids:
        raise RuntimeError("Extraction coverage mismatch")
    manifest = json.loads((root / "path_repository_manifest.json").read_text(encoding="utf-8"))
    if sha256(root / "path_repository.jsonl") != manifest["sha256"]:
        raise RuntimeError("Path repository hash mismatch")
    paths = read_jsonl(root / "path_repository.jsonl")
    if (len(paths) != len({p["id"] for p in paths})
            or len(paths) != manifest["paths_expected"]
            or any(len(p["questions"]) != 2 for p in paths)):
        raise RuntimeError("Path completeness/uniqueness failure")
    graph = json.loads((root / "graph.json").read_text(encoding="utf-8"))
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    resolution_checks = []
    if config.get("method_version") == "source_alias_merge_global_graph_v2":
        from . import prompts
        from .entity_resolution import canonical_mapping, catalog, remap_extractions
        from .graph import assemble
        resolution_root = root / "entity_resolution"
        resolution = json.loads((resolution_root / "manifest.json").read_text(encoding="utf-8"))
        candidates = json.loads((resolution_root / "candidates.json").read_text(encoding="utf-8"))
        mapping = json.loads((resolution_root / "mapping.json").read_text(encoding="utf-8"))
        if (sha256(resolution_root / "candidates.json") != resolution["candidate_sha256"]
                or sha256(resolution_root / "mapping.json") != resolution["mapping_sha256"]):
            raise RuntimeError("Entity resolution artifact hash drift")
        groups = json.loads((root / "inputs/construction_groups.json").read_text(encoding="utf-8"))
        if any(set(group) != {"passage_ids"}
               or not set(group["passage_ids"]) <= source_ids for group in groups):
            raise RuntimeError("Construction groups contain labels or unknown passages")
        extraction_data = {
            p["passage_id"]: json.loads(
                (root / "extractions" / (p["passage_id"] + ".json")).read_text(
                    encoding="utf-8"))["extracted"] for p in corpus}
        fingerprint = signature({
            "extracted": extraction_data, "groups": groups,
            "model_digest": resolution["embedding_model_digest"],
            "threshold": 0.95, "prompt": prompts.MERGE_SYSTEM})
        if fingerprint != resolution["input_fingerprint"]:
            raise RuntimeError("Entity resolution inputs or policy drifted")
        decisions = [decision for file in (resolution_root / "batches").glob("*.json")
                     for decision in json.loads(file.read_text(encoding="utf-8"))["decisions"]]
        expected_pairs = [tuple(pair["keys"]) for pair in candidates["pairs"]]
        actual_pairs = [tuple(pair["keys"]) for pair in decisions]
        if (len(set(expected_pairs)) != len(expected_pairs)
                or sorted(actual_pairs) != sorted(expected_pairs)
                or len(decisions) != resolution["decisions_completed"]):
            raise RuntimeError("Entity confirmation coverage mismatch")
        details, frequencies = catalog(extraction_data)
        expected_mapping = canonical_mapping(
            details, frequencies,
            sorted(tuple(pair["keys"]) for pair in decisions if pair["same_entity"]))
        if expected_mapping != mapping:
            raise RuntimeError("Entity canonical mapping inconsistent with confirmations")
        rebuilt = assemble(corpus, remap_extractions(extraction_data, details, mapping))
        for field in ("entities", "hyperedges", "full_triples", "triples",
                      "bridge_ids", "retained_entity_ids"):
            if rebuilt[field] != graph[field]:
                raise RuntimeError("Graph differs from source extractions and confirmed aliases")
        code = json.loads((root / "code_manifest.json").read_text(encoding="utf-8"))
        for name, expected in code["modules_sha256"].items():
            if sha256(Path(__file__).parent / name) != expected:
                raise RuntimeError("Research implementation changed during the run")
        for name, expected in code.get("supporting_files_sha256", {}).items():
            if sha256(Path(__file__).resolve().parents[1] / name) != expected:
                raise RuntimeError("Supporting prompts, policy or dependency lock changed")
        resolution_checks = [
            "unlabelled_construction_groups", "complete_LLM_alias_confirmation",
            "graph_rebuilt_from_confirmed_mapping", "frozen_implementation_hashes"]
    if len(paths) != graph["stats"]["two_edge_paths"]:
        raise RuntimeError("Graph/path repository counts differ")
    for name, expected_count in (
        ("hyperedges", len(corpus)),
        ("entities", graph["stats"]["sparse_entities"]),
        ("paths", len(paths) * 2),
    ):
        matrix = np.load(root / f"indexes/{name}.npy", mmap_mode="r")
        if matrix.shape != (expected_count, 1024):
            raise RuntimeError("Index shape mismatch")
        for start in range(0, len(matrix), 10000):
            block = matrix[start:start + 10000]
            if not np.isfinite(block).all():
                raise RuntimeError("Nonfinite indexed vector")
            if not np.allclose(np.linalg.norm(block, axis=1), 1, atol=1e-4):
                raise RuntimeError("Indexed vectors not normalized")
    predictions = [json.loads(p.read_text(encoding="utf-8"))
                   for p in (root / "predictions").glob("*.json")]
    if {p["id"] for p in predictions} != set(ids) or len(predictions) != len(ids):
        raise RuntimeError("Prediction coverage/alignment mismatch")
    for prediction in predictions:
        if (prediction["status"] != "completed" or prediction["model"] != MODEL
                or prediction["reasoning_effort"] != EFFORT
                or not set(prediction["retrieved_passage_ids"]) <= source_ids
                or not set(prediction["supporting_passage_ids"])
                <= set(prediction["retrieved_passage_ids"])):
            raise RuntimeError("Prediction identity/evidence boundary failure")
    calls = read_jsonl(root / "api_calls.jsonl") if (root / "api_calls.jsonl").exists() else []
    if any(call.get("model") != MODEL or call.get("provider") != PROVIDER
           or call.get("reasoning_effort") != EFFORT
           or call.get("usage", {}).get("output_tokens_details", {}).get("reasoning_tokens", 0) != 0
           for call in calls):
        raise RuntimeError("Model/effort contract drift")
    evaluation = json.loads((root / "evaluation/metrics.json").read_text(encoding="utf-8"))
    if evaluation["expected"] != len(ids) or evaluation["missing_or_failed"] != 0:
        raise RuntimeError("Evaluation denominator/coverage mismatch")
    if not (root / "runtime_environment.json").exists():
        raise RuntimeError("Runtime and hardware provenance missing")
    if not (root / "resources.jsonl").exists():
        raise RuntimeError("Resource measurements missing")
    result = {
        "status": "passed", "questions": len(ids), "passages": len(corpus),
        "paths": len(paths), "path_questions": len(paths) * 2,
        "api_attempts": len(calls), "original_preserved": preserved,
        "checks": ["source_and_input_hashes", "label_free_inference_questions",
                   "exact_id_alignment", "full_extraction_coverage",
                   "complete_uncapped_path_repository", "normalized_1024d_BGE_M3_indexes",
                   "prediction_evidence_boundaries", "4omini_no_reasoning_tokens",
                   "full_evaluation_denominator"] + resolution_checks,
    }
    write_json(root / "verification.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.run), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
