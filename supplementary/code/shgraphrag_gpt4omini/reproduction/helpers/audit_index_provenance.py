"""Independently reconstruct every index input/row; no API or gold access."""
from __future__ import annotations

import argparse
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from reproduction.client import canonical, write_json
from reproduction.data import sha256

DIGEST = "7907646426070047a77226ac3e684fbbe8410524f7b4a74d02837e43f2146bab"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def input_fingerprint(digest, records):
    """Stream the exact canonical {digest, items, texts} fingerprint encoding."""
    result = hashlib.sha256()
    result.update(b'{"digest":' + canonical(digest).encode("utf-8"))
    for field, position in (("items", 0), ("texts", 1)):
        result.update(b',"' + field.encode("ascii") + b'":[')
        for index, record in enumerate(records()):
            if index:
                result.update(b",")
            result.update(canonical(record[position]).encode("utf-8"))
        result.update(b"]")
    result.update(b"}")
    return result.hexdigest()


def path_records(run):
    with (run / "path_repository.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            path = json.loads(line)
            require(len(path["questions"]) == 2, "Path question coverage differs")
            for index, question in enumerate(path["questions"]):
                yield ({"id": path["id"] + "_" + str(index),
                        "path_id": path["id"]}, question)


def audit_one(run, name, records, count, digest):
    began = time.perf_counter()
    metadata_path = run / "indexes" / (name + ".json")
    matrix_path = run / "indexes" / (name + ".npy")
    metadata_hash = sha256(metadata_path)
    metadata = load(metadata_path)
    require(metadata["model_digest"] == digest, "Index embedding model differs")
    require(metadata["dimensions"] == 1024 and metadata["normalized"] is True,
            "Index embedding configuration differs")
    missing = object()
    observed = 0
    for expected, actual in zip_longest(
            (item for item, _ in records()), metadata["items"], fillvalue=missing):
        require(expected is not missing and actual is not missing,
                "Index metadata row count differs")
        require(expected == actual, "Index ID/order differs at row " + str(observed))
        observed += 1
    require(observed == count, "Expected index input count differs")
    fingerprint = input_fingerprint(digest, records)
    require(metadata["fingerprint"] == fingerprint,
            "Index input text/ID fingerprint differs")
    matrix = np.load(matrix_path, mmap_mode="r")
    require(matrix.shape == (count, 1024) and matrix.dtype == np.float32,
            "Index vector shape/dtype differs")
    minimum_norm, maximum_norm = float("inf"), 0.0
    for start in range(0, count, 10000):
        block = matrix[start:start + 10000]
        require(np.isfinite(block).all(), "Nonfinite vector")
        norms = np.linalg.norm(block, axis=1)
        require(np.allclose(norms, 1, atol=1e-4), "Unnormalized vector")
        minimum_norm = min(minimum_norm, float(norms.min()))
        maximum_norm = max(maximum_norm, float(norms.max()))
    del matrix
    require(sha256(metadata_path) == metadata_hash, "Metadata changed during audit")
    return {
        "status": "passed", "name": name, "vectors": observed, "dimensions": 1024,
        "dtype": "float32", "every_row_id_order_verified": True,
        "every_input_text_fingerprint_verified": True,
        "every_vector_finite_normalized": True,
        "input_fingerprint": fingerprint, "model_digest": digest,
        "metadata_sha256": metadata_hash, "matrix_sha256": sha256(matrix_path),
        "matrix_bytes": matrix_path.stat().st_size,
        "minimum_norm": minimum_norm if count else None,
        "maximum_norm": maximum_norm if count else None,
        "elapsed_s": time.perf_counter() - began,
    }


def audit(run):
    started = time.perf_counter()
    config = load(run / "config.json")
    code = load(run / "code_manifest.json")
    for name, expected in code["modules_sha256"].items():
        require(sha256(ROOT / "reproduction" / name) == expected,
                "Frozen research module differs")
    graph_path = run / "graph.json"
    repository_path = run / "path_repository.jsonl"
    graph_hash = sha256(graph_path)
    repository_hash = sha256(repository_path)
    graph = load(graph_path)
    manifest = load(run / "index_manifest.json")
    repository_manifest = load(run / "path_repository_manifest.json")
    require(manifest["model_digest"] == DIGEST, "Frozen BGE-M3 digest differs")
    require(repository_manifest["sha256"] == repository_hash
            and repository_manifest["paths_expected"]
            == repository_manifest["paths_completed"] == graph["stats"]["two_edge_paths"]
            and repository_manifest["questions"] == 2 * graph["stats"]["two_edge_paths"],
            "Complete path repository prerequisite differs")
    require(config["embedding_dimensions"] == 1024
            and config["vector_dtype"] == "float32", "Frozen index configuration differs")
    hyperedges = list(graph["hyperedges"].values())
    entities = [graph["entities"][eid] for eid in graph["retained_entity_ids"]]
    expected = {
        "hyperedges": (lambda: (
            ({"id": p["id"]}, p["title"] + "\n" + p["text"]) for p in hyperedges),
            len(hyperedges)),
        "entities": (lambda: (
            ({"id": p["id"]}, p["name"]) for p in entities), len(entities)),
        "paths": (lambda: path_records(run), repository_manifest["questions"]),
    }
    evidence = []
    for name, (records, count) in expected.items():
        manifest_key = "path_questions" if name == "paths" else name
        require(manifest[manifest_key] == count, "Index manifest count differs")
        evidence.append(audit_one(run, name, records, count, DIGEST))
    require(sha256(graph_path) == graph_hash
            and sha256(repository_path) == repository_hash,
            "Frozen source artifacts changed during audit")
    result = {
        "status": "passed", "dataset": config["dataset"], "run": str(run),
        "observed_unix": time.time(), "elapsed_s": time.perf_counter() - started,
        "scope": "Complete row/input/file index provenance; no embedding regeneration.",
        "new_model_calls": 0, "new_embedding_calls": 0, "gold_read": False,
        "graph_sha256": graph_hash, "path_repository_sha256": repository_hash,
        "helper_sha256": sha256(Path(__file__)), "indexes": evidence,
        "limitations": [
            "Checks saved input identities/order/fingerprints, all vector norms and file "
            "hashes. It does not independently regenerate every embedding value.",
            "Complete method acceptance also requires source/alias/graph/response audits "
            "and all frozen-question predictions with ID-aligned evaluation.",
        ],
    }
    write_json(run / "index_provenance_audit.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    from reproduction.helpers.storage_paths import checked_run_path
    destination = checked_run_path(args.run)
    print(json.dumps(audit(destination), ensure_ascii=True, indent=2))
