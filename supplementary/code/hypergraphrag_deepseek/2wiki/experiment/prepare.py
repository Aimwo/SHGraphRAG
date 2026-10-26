"""Use independently reconstructed, hash-checked seed42 inputs."""
from __future__ import annotations
import json
import shutil
from .common import PROJECT, ROOT, file_hash, read_json, write_json
from .config import load_config


def prepare():
    dataset = load_config(ready=False)["dataset_id"]
    source = PROJECT / "deepseek" / dataset
    manifest = read_json(source / "manifest.json")
    inputs = ROOT / "inputs"
    inputs.mkdir(exist_ok=True)
    for name, record in manifest["files"].items():
        if file_hash(source / name) != record["sha256"]:
            raise ValueError("Prepared data hash changed: " + name)
        target = inputs / name
        if target.exists() and file_hash(target) != record["sha256"]:
            raise ValueError("Existing experiment input differs: " + name)
        if not target.exists():
            shutil.copyfile(source / name, target)
    write_json(inputs / "manifest.json", manifest)
    return validate_inputs()


def validate_inputs():
    inputs = ROOT / "inputs"
    manifest = read_json(inputs / "manifest.json")
    dataset = load_config(ready=False)["dataset_id"]
    if manifest["dataset_id"] != dataset or manifest["seed"] != 42 or manifest["count"] != 1000:
        raise ValueError("Wrong dataset, seed, or evaluation denominator")
    for name, record in manifest["files"].items():
        if file_hash(inputs / name) != record["sha256"]:
            raise ValueError("Frozen input hash differs: " + name)
    ids = read_json(inputs / "ids.json")
    queries = [json.loads(line) for line in (inputs / "queries.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(set(ids)) != 1000 or [row["id"] for row in queries] != ids or any(set(row) != {"id", "question"} for row in queries):
        raise ValueError("Inference fields or ID alignment differ")
    expected = read_json(ROOT.parents[2] / "protocols/seed42" / dataset / "ids.json")
    if ids != expected:
        raise ValueError("Question IDs differ from the completed experiment")
    return {"passed": True, "dataset": dataset, "questions": 1000, "seed": 42,
            "passages": manifest["unique_candidate_passages"],
            "inputs_sha256": file_hash(inputs / "manifest.json"), "api_calls": 0}


if __name__ == "__main__":
    print(json.dumps(prepare(), indent=2))
