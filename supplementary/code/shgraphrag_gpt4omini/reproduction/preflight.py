from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import requests

from .client import ResponsesClient, parse_json, write_json


def check_original() -> dict:
    root = Path(__file__).resolve().parents[3]
    manifest = json.loads((root / "provenance/source_manifest.json").read_text(encoding="utf-8"))
    changed = []
    prefix = "code/shgraphrag_gpt4omini/"
    for name, record in manifest["files"].items():
        if name.startswith(prefix):
            path = root / name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != record["published_sha256"]:
                changed.append(name)
    return {"source_root": str(root), "source_files_changed": changed,
            "working_tree_status_unchanged": not changed,
            "checked_files": sum(name.startswith(prefix) for name in manifest["files"])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    preserved = check_original()
    if preserved["source_files_changed"] or not preserved["working_tree_status_unchanged"]:
        raise RuntimeError("Original checkout differs from initial snapshot")
    session = requests.Session()
    session.trust_env = False
    tags = session.get("http://127.0.0.1:11434/api/tags", timeout=10).json()
    model = next(m for m in tags["models"] if m["name"] == "bge-m3:latest")
    started = time.perf_counter()
    response = session.post("http://127.0.0.1:11434/api/embed",
                            json={"model": "bge-m3:latest", "input": [
                                "Who directed the film?",
                                "The film was directed by Alice."]},
                            timeout=120)
    response.raise_for_status()
    array = np.array(response.json()["embeddings"], dtype=np.float32)
    if array.shape != (2, 1024) or not np.isfinite(array).all():
        raise RuntimeError("Unexpected BGE-M3 dimension or nonfinite embedding")
    norms = np.linalg.norm(array, axis=1)
    if (norms < 0.1).any():
        raise RuntimeError("Invalid zero embedding")
    embedding = {"endpoint": "http://127.0.0.1:11434", "model": model,
                 "shape": list(array.shape), "norms": norms.tolist(),
                 "elapsed_s": time.perf_counter() - started}
    client = ResponsesClient(args.output, concurrency=1)
    completion = client.call(
        stage="preflight", item_id="4omini-json-contract",
        system="Return only JSON matching the required schema.",
        prompt='Return {"ok":true}.', max_tokens=64,
        response_schema={"type": "object", "properties": {
            "ok": {"type": "boolean"}}, "required": ["ok"],
            "additionalProperties": False})
    if parse_json(completion["text"]) != {"ok": True}:
        raise RuntimeError("GPT-4o-mini contract probe did not return OK")
    result = {"original_preserved": preserved, "embedding": embedding,
              "openrouter": completion, "status": "passed"}
    write_json(args.output / "preflight.json", result)
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
