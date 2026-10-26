"""Answer evaluation aligned to exact frozen IDs; gold is never used by inference."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import string

from .client import write_json


def normalize(text: str) -> str:
    text = "".join(c for c in text.lower() if c not in set(string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def answer_scores(prediction: str, target: str, *, hotpot: bool = True) -> tuple:
    pred, gold = normalize(prediction), normalize(target)
    em = float(pred == gold)
    if hotpot and (pred in ("yes", "no", "noanswer")
                   or gold in ("yes", "no", "noanswer")) and pred != gold:
        return em, 0.0
    a, b = pred.split(), gold.split()
    if not a or not b:
        return em, (0.0 if hotpot else float(a == b))
    common = sum((Counter(a) & Counter(b)).values())
    return em, (2 * common / (len(a) + len(b)) if common else 0.0)


def passage_id(title: str, text: str) -> str:
    return "passage_" + hashlib.sha256((title + "\0" + text).encode()).hexdigest()


def gold_passages(row: dict, dataset: str) -> set[str]:
    if dataset == "musique":
        return {passage_id(p["title"], p["paragraph_text"])
                for p in row["paragraphs"] if p["is_supporting"]}
    support_titles = {fact[0] for fact in row["supporting_facts"]}
    return {passage_id(title, "\n".join(sentences))
            for title, sentences in row["context"] if title in support_titles}


def evaluate_rows(gold: list[dict], predictions: dict[str, dict], dataset: str,
                  aliases: dict | None = None) -> dict:
    ids = [r.get("_id") or r["id"] for r in gold]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate gold IDs")
    if set(predictions) - set(ids):
        raise ValueError("Prediction IDs outside the frozen evaluation set")
    totals = Counter()
    per_sample = []
    for row, qid in zip(gold, ids):
        pred = predictions.get(qid)
        if pred is None or pred.get("status") != "completed":
            per_sample.append({"id": qid, "status": "missing_or_failed",
                               "em": 0.0, "f1": 0.0})
            continue
        text = pred["answer"]
        em, f1 = answer_scores(text, row["answer"])
        references = [row["answer"]]
        if dataset == "musique":
            references += row.get("answer_aliases", [])
        elif dataset == "2wiki":
            item = (aliases or {}).get(row.get("answer_id"), {})
            references += item.get("aliases", []) + item.get("demonyms", [])
        native = [answer_scores(text, ref, hotpot=dataset != "musique")
                  for ref in references]
        expected = gold_passages(row, dataset)
        retrieved = set(pred.get("retrieved_passage_ids", []))
        chosen = set(pred.get("supporting_passage_ids", []))
        recall = len(expected & retrieved) / len(expected) if expected else 0.0
        supporting_recall = len(expected & chosen) / len(expected) if expected else 0.0
        supporting_precision = len(expected & chosen) / len(chosen) if chosen else 0.0
        supporting_f1 = (2 * supporting_precision * supporting_recall
                         / (supporting_precision + supporting_recall)
                         if supporting_precision + supporting_recall else 0.0)
        metrics = {"em": em, "f1": f1,
                   "native_em": max(v[0] for v in native),
                   "native_f1": max(v[1] for v in native),
                   "retrieval_passage_recall": recall,
                   "retrieval_all_supports": float(expected <= retrieved),
                   "supporting_passage_f1": supporting_f1}
        totals.update(metrics)
        totals["completed"] += 1
        totals["direct"] += pred.get("route") == "direct"
        totals["nonempty"] += bool(text.strip())
        per_sample.append({"id": qid, "prediction": text, "gold": row["answer"],
                           "route": pred.get("route"), **metrics})
    count = len(gold)
    return {
        "dataset": dataset, "expected": count,
        "completed": totals["completed"], "missing_or_failed": count - totals["completed"],
        "primary_protocol": "HotpotQA_single_reference_answer_EM_token_F1",
        "primary_em": 100 * totals["em"] / count,
        "primary_f1": 100 * totals["f1"] / count,
        "native_alias_em": 100 * totals["native_em"] / count,
        "native_alias_f1": 100 * totals["native_f1"] / count,
        "retrieval_passage_recall": 100 * totals["retrieval_passage_recall"] / count,
        "retrieval_all_supports": 100 * totals["retrieval_all_supports"] / count,
        "supporting_passage_f1": 100 * totals["supporting_passage_f1"] / count,
        "support_metric_note": "Passage-level diagnostic, not official sentence-level support F1.",
        "direct_count": totals["direct"],
        "direct_fraction": totals["direct"] / count,
        "nonempty_answers": totals["nonempty"], "per_sample": per_sample,
    }


def evaluate_run(root: Path) -> dict:
    data = json.loads((root / "data_manifest.json").read_text(encoding="utf-8"))
    gold = json.loads((root / "evaluation/gold.json").read_text(encoding="utf-8"))
    aliases_path = root / "evaluation/aliases.json"
    aliases = (json.loads(aliases_path.read_text(encoding="utf-8"))
               if aliases_path.exists() else {})
    predictions = {}
    for file in (root / "predictions").glob("*.json"):
        item = json.loads(file.read_text(encoding="utf-8"))
        if item["id"] in predictions:
            raise ValueError("Duplicate prediction ID")
        predictions[item["id"]] = item
    result = evaluate_rows(gold, predictions, data["dataset"], aliases)
    write_json(root / "evaluation/metrics.json", result)
    write_json(root / "predictions.json", {
        "answer": {key: value["answer"] for key, value in predictions.items()
                   if value["status"] == "completed"},
    })
    return {k: v for k, v in result.items() if k != "per_sample"}
