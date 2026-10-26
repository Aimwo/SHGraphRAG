"""Source-style alias confirmation, confined to unlabelled construction groups.

The historical builder uses BGE name cosine > .95 OR name containment, then
asks the LLM to confirm batches of ten pairs. Its question-suffixed identities
are not retained: this reproduction builds one global graph, and never filters
retrieval by question membership. Positive pairs are reconciled transitively
before choosing the most frequent source name as canonical.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from . import prompts
from .client import signature, write_json
from .data import sha256


MERGE_THRESHOLD = 0.95
MERGE_BATCH = 10


def catalog(extracted: dict[str, dict]) -> tuple[dict, Counter]:
    details, frequencies = {}, Counter()
    for data in extracted.values():
        for key, entity in data["entities"].items():
            details.setdefault(key, entity)
            frequencies[key] += 1
    return details, frequencies


def candidate_pairs(groups: list[dict], extracted: dict, details: dict,
                    vectors: np.ndarray):
    positions = {key: i for i, key in enumerate(details)}
    seen = set()
    for group in groups:
        keys = sorted({key for pid in group["passage_ids"]
                       for key in extracted[pid]["entities"]})
        if len(keys) < 2:
            continue
        matrix = vectors[[positions[key] for key in keys]]
        similarities = matrix @ matrix.T
        for i, j in combinations(range(len(keys)), 2):
            a, b = keys[i], keys[j]
            pair = (a, b)
            if pair in seen:
                continue
            if (float(similarities[i, j]) > MERGE_THRESHOLD
                    or a in b or b in a):
                seen.add(pair)
                yield {"keys": [a, b], "cosine": float(similarities[i, j]),
                       "substring": a in b or b in a}


def canonical_mapping(details: dict, frequencies: Counter,
                      confirmed: list[tuple[str, str]]) -> dict[str, str]:
    parent = {key: key for key in details}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for a, b in confirmed:
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a
    components = {}
    for key in details:
        components.setdefault(find(key), []).append(key)
    # The source chooses a more frequent name; break ties deterministically.
    order = {key: i for i, key in enumerate(details)}
    result = {}
    for keys in components.values():
        canonical = min(keys, key=lambda k: (-frequencies[k], order[k], k))
        result.update({key: canonical for key in keys})
    return result


def remap_extractions(extracted: dict, details: dict, mapping: dict) -> dict:
    remapped = {}
    for pid, data in extracted.items():
        entities = {}
        for key, item in data["entities"].items():
            target = mapping[key]
            bucket = entities.setdefault(target, {
                "name": details[target]["name"], "type": details[target]["type"],
                "description": "", "descriptions": [], "aliases": []})
            if item["description"] not in bucket["descriptions"]:
                bucket["descriptions"].append(item["description"])
            if item["name"] not in bucket["aliases"]:
                bucket["aliases"].append(item["name"])
            bucket["description"] = "\n".join(bucket["descriptions"])
        relations = []
        for rel in data["relations"]:
            head, tail = mapping[rel["head"]], mapping[rel["tail"]]
            if head != tail:
                relations.append({**rel, "head": head, "tail": tail})
        remapped[pid] = {"entities": entities, "relations": relations}
    return remapped


def resolve(runner, extracted: dict, groups: list[dict], parallel_work, batches):
    details, frequencies = catalog(extracted)
    vectors = runner.embeddings().encode(
        [item["name"] for item in details.values()], stage="entity_merge_names")
    candidates = list(candidate_pairs(groups, extracted, details, vectors))
    root = runner.root / "entity_resolution"
    fingerprint = signature({
        "extracted": extracted, "groups": groups,
        "model_digest": runner.embeddings().digest,
        "threshold": MERGE_THRESHOLD, "prompt": prompts.MERGE_SYSTEM})
    candidate_path = root / "candidates.json"
    write_json(candidate_path, {
        "input_fingerprint": fingerprint, "pairs": candidates})

    def one(batch):
        payload = []
        for i, candidate in enumerate(batch):
            a, b = candidate["keys"]
            payload.append({"id": f"p{i}", "entities": [
                {"name": details[k]["name"], "type": details[k]["type"],
                 "description": details[k]["description"]} for k in (a, b)]})
        expected = signature([prompts.MERGE_SYSTEM, payload])
        destination = root / "batches" / f"{expected}.json"
        if destination.exists():
            item = json.loads(destination.read_text(encoding="utf-8"))
            if item["input_fingerprint"] != expected:
                raise RuntimeError("Entity confirmation input fingerprint changed")
            return item["decisions"]
        ids = {item["id"] for item in payload}

        def validate(value):
            rows = value["pairs"]
            if (len(rows) != len(ids) or {r["id"] for r in rows} != ids
                    or any(not isinstance(r["same_entity"], bool) for r in rows)):
                raise ValueError("Invalid entity-pair confirmation coverage")

        value, response = runner.structured(
            "entity_merge", expected, prompts.MERGE_SYSTEM,
            {"pairs": payload}, validate, max_tokens=2048,
            schema=prompts.obj({"pairs": {
                "type": "array", "minItems": len(batch), "maxItems": len(batch),
                "items": prompts.obj({
                    "id": {"type": "string", "enum": sorted(ids)},
                    "same_entity": prompts.BOOL})}}))
        decisions = [{"keys": batch[int(row["id"][1:])]["keys"],
                      "same_entity": row["same_entity"]} for row in value["pairs"]]
        write_json(destination, {
            "input_fingerprint": expected,
            "request_sha256": response["request_sha256"], "decisions": decisions})
        return decisions

    decisions, completed = [], 0

    def done(rows):
        nonlocal completed
        decisions.extend(rows)
        completed += len(rows)
        runner.progress("entity_merge", completed=completed, total=len(candidates))

    parallel_work(batches(candidates, MERGE_BATCH), one,
                  workers=runner.workers, on_done=done)
    confirmed = sorted(tuple(item["keys"]) for item in decisions
                       if item["same_entity"])
    mapping = canonical_mapping(details, frequencies, confirmed)
    write_json(root / "mapping.json", mapping)
    write_json(root / "manifest.json", {
        "input_fingerprint": fingerprint,
        "embedding_model_digest": runner.embeddings().digest,
        "input_entities": len(details), "construction_groups": len(groups),
        "candidate_pairs": len(candidates), "confirmed_pairs": len(confirmed),
        "decisions_completed": completed,
        "canonical_entities": len(set(mapping.values())),
        "names_merged": sum(key != value for key, value in mapping.items()),
        "threshold": MERGE_THRESHOLD, "batch_size": MERGE_BATCH,
        "candidate_sha256": sha256(candidate_path),
        "mapping_sha256": sha256(root / "mapping.json"),
        "scope": "unlabelled_candidate_groups_construction_only",
        "canonical_policy": "global_passage_frequency_then_first_source",
        "description_policy": "all_source_descriptions_preserved_no_new_summary_call",
    })
    return remap_extractions(extracted, details, mapping)
