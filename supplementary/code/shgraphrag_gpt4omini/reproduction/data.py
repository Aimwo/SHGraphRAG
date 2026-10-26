"""Prepare label-free inference inputs and strictly separated evaluation gold."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .client import write_json


DATASETS = ("hotpotqa", "2wiki", "musique")
SAMPLE = "sample_1000_seed20261020"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def prepare(source: Path, output: Path, dataset: str, limit: int) -> dict:
    if dataset not in DATASETS or not 0 < limit <= 1000:
        raise ValueError("Expected one frozen dataset and 1..1000 questions")
    manifest_path = source / "dataset/paper_ready/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["datasets"][dataset]
    mappings = {
        "sample_ids": entry["sample_ids_sha256"],
        "sample_gold": entry["sample_gold_sha256"],
        "sample_parquet": entry["sample_parquet_sha256"],
    }
    files = {}
    for field, expected in mappings.items():
        path = source / entry[field]
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"Frozen dataset hash mismatch: {field}")
        files[field] = {"path": str(path), "sha256": actual}
    corpus_info = entry["candidate_corpus"]
    for field in ("corpus", "question_passages"):
        path = source / corpus_info[field]
        actual = sha256(path)
        if actual != corpus_info[field + "_sha256"]:
            raise RuntimeError(f"Frozen corpus hash mismatch: {field}")
        files[field] = {"path": str(path), "sha256": actual}
    ids = json.loads(Path(files["sample_ids"]["path"]).read_text(encoding="utf-8"))
    gold_path = Path(files["sample_gold"]["path"])
    native = (read_jsonl(gold_path) if dataset == "musique" else
              json.loads(gold_path.read_text(encoding="utf-8")))
    if len(ids) != 1000 or len(set(ids)) != 1000:
        raise RuntimeError("Frozen ID set is not exactly 1000 unique questions")
    if [(r.get("_id") or r["id"]) for r in native] != ids:
        raise RuntimeError("Native gold and frozen IDs differ in order")
    selected_ids = ids[:limit]
    selected_gold = native[:limit]
    # Membership is used ONLY to choose the pilot corpus, never at retrieval.
    membership = read_jsonl(Path(files["question_passages"]["path"]))
    selected_passages = {pid for item in membership
                         if item["question_id"] in set(selected_ids)
                         for pid in item["passage_ids"]}
    corpus = read_jsonl(Path(files["corpus"]["path"]))
    if limit < 1000:
        corpus = [p for p in corpus if p["passage_id"] in selected_passages]
    if set(p["passage_id"] for p in corpus) != selected_passages:
        raise RuntimeError("Selected candidate-corpus membership mismatch")
    for passage in corpus:
        if passage["text"] != "\n".join(passage["sentences"]):
            raise RuntimeError("Passage sentence/text mismatch")
    inputs = output / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    questions = [{"id": r.get("_id") or r["id"], "question": r["question"]}
                 for r in selected_gold]
    write_json(inputs / "questions.json", questions)
    write_json(inputs / "ids.json", selected_ids)
    # The supplied builder resolves aliases within each candidate document
    # group. Preserve that *unlabelled construction scope*, without attaching
    # a question ID, answer, support label, or membership filter to the graph.
    groups = sorted({tuple(sorted(item["passage_ids"])) for item in membership
                     if item["question_id"] in set(selected_ids)})
    write_json(inputs / "construction_groups.json",
               [{"passage_ids": list(group)} for group in groups])
    with (inputs / "corpus.jsonl").open("w", encoding="utf-8") as stream:
        for passage in corpus:
            stream.write(json.dumps(passage, ensure_ascii=False) + "\n")
    write_json(output / "evaluation/gold.json", selected_gold)
    if dataset == "2wiki":
        alias_path = source / entry["answer_aliases"]
        if sha256(alias_path) != entry["answer_aliases_sha256"]:
            raise RuntimeError("2Wiki aliases hash mismatch")
        wanted = {r.get("answer_id") for r in selected_gold}
        aliases = {row["Q_id"]: row for row in read_jsonl(alias_path)
                   if row["Q_id"] in wanted}
        write_json(output / "evaluation/aliases.json", aliases)
        files["answer_aliases"] = {"path": str(alias_path), "sha256": sha256(alias_path)}
    result = {
        "dataset": dataset, "questions": len(questions),
        "passages": len(corpus), "seed": 20261020,
        "sample_status": "new_revision_sample_not_original_paper_sample",
        "scope": "union_of_selected_questions_candidate_passages",
        "retrieval_filter": "none_global_union",
        "entity_merge_scope": "unlabelled_candidate_groups_construction_only",
        "selection": f"first_{limit}_frozen_ids_in_saved_sample_order",
        "source_files": files,
        "inputs_sha256": {p.name: sha256(p) for p in inputs.iterdir()},
    }
    write_json(output / "data_manifest.json", result)
    return result
