"""Offline, exact-ID answer scoring; never imported by inference transports."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
import string

from .transport import file_hash, write_json


def normalize(text):
    text = "".join(c for c in text.lower() if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def scores(prediction, target, hotpot=True):
    pred, gold = normalize(prediction), normalize(target)
    em = float(pred == gold)
    if hotpot and (pred in ("yes", "no", "noanswer")
                   or gold in ("yes", "no", "noanswer")) and pred != gold:
        return em, 0.0
    a, b = pred.split(), gold.split()
    if not a or not b:
        return em, 0.0 if hotpot else float(a == b)
    shared = sum((Counter(a) & Counter(b)).values())
    return em, 2 * shared / (len(a) + len(b)) if shared else 0.0


def evaluate_run(root, project_root, entry):
    root, project_root = Path(root), Path(project_root)
    plan = json.loads((root / "data_manifest.json").read_text(encoding="utf-8"))
    source = entry["sources"]["native_gold_evaluation_only"]
    path = project_root / source["path"]
    if file_hash(path) != source["sha256"]:
        raise RuntimeError("Gold source hash mismatch")
    gold = ([json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            if entry["dataset"] == "musique" else
            json.loads(path.read_text(encoding="utf-8")))
    chosen = set(plan["ids"])
    gold = [r for r in gold if (r.get("_id") or r["id"]) in chosen]
    if [r.get("_id") or r["id"] for r in gold] != plan["ids"]:
        raise RuntimeError("Gold/frozen ID order mismatch")
    aliases = {}
    if entry["dataset"] == "2wiki":
        manifest = json.loads((project_root / "dataset/"
                               "paper_ready/manifest.json").read_text(encoding="utf-8"))
        data = manifest["datasets"]["2wiki"]
        alias_path = project_root / data["answer_aliases"]
        if file_hash(alias_path) != data["answer_aliases_sha256"]:
            raise RuntimeError("2Wiki alias hash mismatch")
        aliases = {row["Q_id"]: row for row in map(json.loads,
                   alias_path.read_text(encoding="utf-8").splitlines())}
    predictions = {}
    for path in (root / "predictions").glob("*.json"):
        pred = json.loads(path.read_text(encoding="utf-8"))
        if pred["id"] in predictions:
            raise RuntimeError("Duplicate prediction ID")
        predictions[pred["id"]] = pred
    if set(predictions) - chosen:
        raise RuntimeError("Unexpected prediction ID")
    totals, details = Counter(), []
    corpus_path = project_root / entry["sources"]["candidate_corpus"]["path"]
    corpus = [json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines()]
    by_text = {(p["title"], p["text"]): p["passage_id"] for p in corpus}
    for row in gold:
        qid = row.get("_id") or row["id"]
        pred = predictions.get(qid, {})
        output = {"id": qid, "status": pred.get("status", "missing")}
        if pred.get("status") != "completed":
            details.append({**output, "em": 0.0, "f1": 0.0})
            continue
        answer = pred["answer"]
        em, f1 = scores(answer, row["answer"])
        references = [row["answer"]]
        if entry["dataset"] == "musique":
            references += row.get("answer_aliases", [])
            pairs = [(p["title"], p["paragraph_text"]) for p in row["paragraphs"]
                     if p["is_supporting"]]
        else:
            if entry["dataset"] == "2wiki":
                item = aliases.get(row.get("answer_id"), {})
                references += item.get("aliases", []) + item.get("demonyms", [])
            titles = {item[0] for item in row["supporting_facts"]}
            pairs = [(title, "\n".join(sentences)) for title, sentences in row["context"]
                     if title in titles]
        expected = {by_text[pair] for pair in pairs}
        native = [scores(answer, ref, hotpot=entry["dataset"] != "musique")
                  for ref in references]
        metrics = {"em": em, "f1": f1, "native_em": max(v[0] for v in native),
                   "native_f1": max(v[1] for v in native)}
        retrieved = pred["retrieved_passage_ids"]
        for k in (2, 5, 10):
            selected = set(retrieved[:k])
            metrics[f"passage_recall_at_{k}"] = len(selected & expected) / len(expected)
            metrics[f"all_supports_at_{k}"] = float(expected <= selected)
        totals.update(metrics)
        totals["completed"] += 1
        totals["nonempty"] += bool(answer.strip())
        details.append({**output, "answer": answer, "gold": row["answer"], **metrics})
    n = len(gold)
    result = {"dataset": entry["dataset"], "expected": n,
              "completed": totals["completed"], "missing_or_failed": n - totals["completed"],
              "primary_protocol": "HotpotQA_single_reference_answer_EM_token_F1",
              "primary_em": totals["em"] * 100 / n,
              "primary_f1": totals["f1"] * 100 / n,
              "native_alias_em": totals["native_em"] * 100 / n,
              "native_alias_f1": totals["native_f1"] * 100 / n,
              "nonempty_answers": totals["nonempty"],
              "retrieval_order": "global_branch_sources_then_local_deduplicated",
              "per_sample": details}
    for key in totals:
        if key.startswith(("passage_", "all_supports_")):
            result[key] = 100 * totals[key] / n
    write_json(root / "evaluation/metrics.json", result)
    write_json(root / "predictions.json", {"answer": {
        key: value["answer"] for key, value in predictions.items()
        if value["status"] == "completed"}})
    return {k: v for k, v in result.items() if k != "per_sample"}
