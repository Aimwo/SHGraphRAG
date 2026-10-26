"""Evaluation-only gold access, exact frozen IDs and an invariant denominator."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import re
import string

from .common import ROOT, read_json, read_jsonl, write_json


def normalize(text):
    value = "".join(c for c in text.lower() if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", value).split())


def answer_scores(prediction, reference):
    pred, gold = normalize(prediction), normalize(reference)
    em = float(pred == gold)
    a, b = pred.split(), gold.split()
    if not a or not b:
        return em, float(a == b)
    shared = sum((Counter(a) & Counter(b)).values())
    return em, 2 * shared / (len(a) + len(b)) if shared else 0.0


def score_records(ids, gold_rows, predictions):
    if len(set(ids)) != len(ids):
        raise ValueError("Frozen IDs contain duplicates")
    gold_by_id = {row["id"]: row for row in gold_rows}
    if len(gold_by_id) != len(gold_rows) or set(gold_by_id) != set(ids):
        raise ValueError("Gold/frozen ID sets differ")
    if set(predictions) - set(ids):
        raise ValueError("Unexpected prediction IDs")
    details = []
    for qid in ids:
        row, prediction = gold_by_id[qid], predictions.get(qid)
        available = bool(prediction and prediction.get("status") == "completed")
        answer = prediction.get("answer", "") if available else ""
        references = [row["answer"], *row.get("answer_aliases", [])]
        em, f1 = answer_scores(answer, row["answer"]) if available else (0.0, 0.0)
        aliases = ([answer_scores(answer, ref) for ref in references] if available
                   else [(0.0, 0.0)])
        support_ids = {hashlib.sha256(
            (p["title"] + "\n" + p["paragraph_text"]).encode("utf-8")).hexdigest()
            for p in row["paragraphs"] if p["is_supporting"]}
        # Retrieval remains observable if answer generation subsequently fails.
        retrieved = prediction.get("retrieved_passage_ids", []) if prediction else []
        metrics = {"single_reference_em": em, "single_reference_f1": f1,
                   "native_alias_em": max(x[0] for x in aliases),
                   "native_alias_f1": max(x[1] for x in aliases)}
        for k in (2, 5, 10):
            selected = set(retrieved[:k])
            metrics[f"passage_recall_at_{k}"] = (
                len(selected & support_ids) / len(support_ids) if support_ids else None)
            metrics[f"all_supports_at_{k}"] = float(support_ids <= selected)
        details.append({
            "id": qid, "hop_group": qid.split("__", 1)[0],
            "status": prediction.get("status", "missing") if prediction else "missing",
            "answer": answer, "gold_answer": row["answer"], "gold_aliases": references,
            "answer_outcome": prediction.get("answer_outcome") if prediction else "missing",
            **metrics,
        })
    groups = defaultdict(list)
    for row in details:
        groups[row["hop_group"]].append(row)

    def aggregate(rows):
        output = {"denominator": len(rows)}
        for key in ("single_reference_em", "single_reference_f1",
                    "native_alias_em", "native_alias_f1"):
            output[key] = 100 * sum(r[key] for r in rows) / len(rows)
        for k in (2, 5, 10):
            key = f"passage_recall_at_{k}"
            known = [r[key] for r in rows if r[key] is not None]
            output[key] = 100 * sum(known) / len(known) if known else None
            output[f"all_supports_at_{k}"] = (
                100 * sum(r[f"all_supports_at_{k}"] for r in rows) / len(rows))
        return output

    return {
        "dataset": "MuSiQue-Ans v1.0 development", **aggregate(details),
        "completed": sum(r["status"] == "completed" for r in details),
        "missing_or_failed": sum(r["status"] != "completed" for r in details),
        "nonempty_answers": sum(bool(r["answer"].strip()) for r in details),
        "native_protocol": "MuSiQue normalization; max EM and token-F1 over native answer aliases",
        "retrieval_order": "global branch sources then local branch sources, deduplicated",
        "hop_groups": {group: aggregate(rows) for group, rows in groups.items()},
        "answer_outcomes": dict(Counter(r["answer_outcome"] for r in details)),
        "per_sample": details,
    }


def evaluate(run_root=None, input_root=None):
    run_root, input_root = (ROOT / "run" if run_root is None else run_root,
                            ROOT / "inputs" if input_root is None else input_root)
    ids = read_json(input_root / "ids.json")
    gold = read_jsonl(input_root / "gold.jsonl")
    predictions = {}
    for path in (run_root / "predictions").glob("*.json"):
        value = read_json(path)
        if value["id"] in predictions or path.stem != value["id"]:
            raise ValueError("Duplicate prediction ID or filename mismatch")
        predictions[value["id"]] = value
    metrics = score_records(ids, gold, predictions)
    if metrics["denominator"] != 1000:
        raise ValueError("Formal evaluation requires all 1000 frozen IDs")
    write_json(run_root / "evaluation/metrics.json", metrics)
    write_json(run_root / "evaluation/answer_predictions.json", [
        {"id": qid,
         "answer": (predictions[qid].get("answer", "")
                    if predictions.get(qid, {}).get("status") == "completed" else ""),
         "status": predictions.get(qid, {}).get("status", "missing")}
        for qid in ids])
    return {k: v for k, v in metrics.items() if k != "per_sample"}


if __name__ == "__main__":
    print(evaluate())
