"""Native answer scoring over every frozen ID; evaluation-only gold access."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import re
import string

from .common import ROOT, read_json, read_jsonl, write_json


def normalize(text):
    value = "".join(c for c in text.lower() if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", value).split())


def answer_scores(prediction, reference, dataset="musique"):
    pred, gold = normalize(prediction), normalize(reference)
    em = float(pred == gold)
    if dataset != "musique":
        boolean = {"yes", "no", "noanswer"}
        if (pred in boolean or gold in boolean) and pred != gold:
            return em, 0.0
    a, b = pred.split(), gold.split()
    if not a or not b:
        return em, float(a == b) if dataset == "musique" else 0.0
    shared = sum((Counter(a) & Counter(b)).values())
    return em, 2 * shared / (len(a) + len(b)) if shared else 0.0


def identifier(row):
    return row.get("_id") or row["id"]


def score_records(ids, gold_rows, predictions, *, dataset=None, aliases=None):
    # Automatic MuSiQue detection preserves the established transport/algorithm
    # regression fixtures. Formal scoring always passes its frozen dataset ID.
    dataset = dataset or ("musique" if gold_rows and "paragraphs" in gold_rows[0] else "hotpotqa")
    if dataset not in ("musique", "hotpotqa", "2wiki"):
        raise ValueError("Unknown native scoring protocol")
    if len(set(ids)) != len(ids):
        raise ValueError("Frozen IDs contain duplicates")
    gold_by_id = {identifier(row): row for row in gold_rows}
    if len(gold_by_id) != len(gold_rows) or set(gold_by_id) != set(ids):
        raise ValueError("Gold/frozen ID sets differ")
    if set(predictions) - set(ids):
        raise ValueError("Unexpected prediction IDs")
    aliases = aliases or {}
    details = []
    for qid in ids:
        row, prediction = gold_by_id[qid], predictions.get(qid)
        available = bool(prediction and prediction.get("status") == "completed")
        answer = prediction.get("answer", "") if available else ""
        references = [row["answer"]]
        if dataset == "musique":
            references += row.get("answer_aliases", [])
            texts = [p["title"] + "\n" + p["paragraph_text"] for p in row["paragraphs"]
                     if p["is_supporting"]]
            group = qid.split("__", 1)[0]
        else:
            if dataset == "2wiki":
                native = aliases.get(row.get("answer_id"), {})
                references += native.get("aliases", []) + native.get("demonyms", [])
            titles = {item[0] for item in row["supporting_facts"]}
            texts = [title + "\n" + "\n".join(sentences)
                     for title, sentences in row["context"] if title in titles]
            group = row["type"]
        support_ids = {hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts}
        em, f1 = answer_scores(answer, row["answer"], dataset) if available else (0.0, 0.0)
        pairs = ([answer_scores(answer, ref, dataset) for ref in references] if available
                 else [(0.0, 0.0)])
        retrieved = prediction.get("retrieved_passage_ids", []) if prediction else []
        metrics = {
            "single_reference_em": em, "single_reference_f1": f1,
            "native_alias_em": max(pair[0] for pair in pairs),
            "native_alias_f1": max(pair[1] for pair in pairs),
        }
        for k in (2, 5, 10):
            selected = set(retrieved[:k])
            metrics[f"passage_recall_at_{k}"] = (
                len(selected & support_ids) / len(support_ids) if support_ids else None)
            metrics[f"all_supports_at_{k}"] = float(support_ids <= selected) if support_ids else None
        details.append({
            "id": qid, "hop_group": group,
            "status": prediction.get("status", "missing") if prediction else "missing",
            "answer": answer, "gold_answer": row["answer"], "gold_aliases": references,
            "answer_outcome": prediction.get("answer_outcome") if prediction else "missing",
            **metrics,
        })

    def aggregate(entries):
        output = {"denominator": len(entries)}
        for key in ("single_reference_em", "single_reference_f1", "native_alias_em", "native_alias_f1"):
            output[key] = 100 * sum(entry[key] for entry in entries) / len(entries)
        for k in (2, 5, 10):
            for key in (f"passage_recall_at_{k}", f"all_supports_at_{k}"):
                known = [entry[key] for entry in entries if entry[key] is not None]
                output[key] = 100 * sum(known) / len(known) if known else None
        return output

    groups = defaultdict(list)
    for detail in details:
        groups[detail["hop_group"]].append(detail)
    return {
        "dataset": dataset, **aggregate(details),
        "completed": sum(entry["status"] == "completed" for entry in details),
        "missing_or_failed": sum(entry["status"] != "completed" for entry in details),
        "nonempty_answers": sum(bool(entry["answer"].strip()) for entry in details),
        "native_protocol": (
            "MuSiQue native normalization and answer aliases" if dataset == "musique"
            else "Native Hotpot/2Wiki boolean-aware EM/token-F1; 2Wiki official answer-ID aliases/demonyms"),
        "retrieval_order": "global branch sources then local branch sources, deduplicated",
        "retrieval_gold_scope": "official supporting titles present in native candidate contexts",
        "hop_groups": {group: aggregate(entries) for group, entries in groups.items()},
        "answer_outcomes": dict(Counter(entry["answer_outcome"] for entry in details)),
        "per_sample": details,
    }


def evaluate(run_root=None, input_root=None):
    run_root = ROOT / "run" if run_root is None else run_root
    input_root = ROOT / "inputs" if input_root is None else input_root
    manifest = read_json(input_root / "manifest.json")
    ids, gold = read_json(input_root / "ids.json"), read_jsonl(input_root / "gold.jsonl")
    predictions = {}
    for path in (run_root / "predictions").glob("*.json"):
        value = read_json(path)
        if value["id"] in predictions or path.stem != value["id"]:
            raise ValueError("Duplicate prediction ID or filename mismatch")
        predictions[value["id"]] = value
    aliases = {row["Q_id"]: row for row in read_jsonl(input_root / "answer_aliases.jsonl")}
    metrics = score_records(ids, gold, predictions, dataset=manifest["dataset_id"], aliases=aliases)
    if metrics["denominator"] != 1000:
        raise ValueError("Formal evaluation requires all 1000 frozen IDs")
    write_json(run_root / "evaluation/metrics.json", metrics)
    write_json(run_root / "evaluation/answer_predictions.json", [
        {"id": qid,
         "answer": (predictions[qid].get("answer", "")
                    if predictions.get(qid, {}).get("status") == "completed" else ""),
         "status": predictions.get(qid, {}).get("status", "missing")} for qid in ids])
    return {key: value for key, value in metrics.items() if key != "per_sample"}


if __name__ == "__main__":
    print(evaluate())
