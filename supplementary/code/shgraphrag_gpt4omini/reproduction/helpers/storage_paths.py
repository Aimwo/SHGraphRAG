"""Accept only isolated runs or their explicitly recorded storage junctions."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "reproduction/provenance/experiment_storage.json"


def checked_run_path(path):
    candidate = Path(path).absolute()
    resolved = candidate.resolve()
    runs = ROOT / "reproduction/runs"
    if resolved.is_relative_to(runs):
        return candidate
    if not REGISTRY.is_file():
        raise ValueError("Run is outside isolated storage and no explicit mapping exists")
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    matches = []
    for entry in registry["mappings"]:
        canonical = Path(entry["canonical_path"])
        target = Path(entry["target_path"])
        if (canonical.is_relative_to(runs)
                and resolved == target.resolve()
                and canonical.resolve() == target.resolve()
                and canonical.absolute() != canonical.resolve()
                and entry["status"] == "verified_junction"):
            matches.append(canonical)
    if len(matches) != 1:
        raise ValueError("Run does not match exactly one verified experiment storage mapping")
    return matches[0]
