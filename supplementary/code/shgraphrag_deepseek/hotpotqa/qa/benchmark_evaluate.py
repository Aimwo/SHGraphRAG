"""Strict frozen-ID answer scoring using unchanged native metric functions."""

import ast
from collections import Counter
import json
import os
from pathlib import Path
import re
import string

ROOT = Path(__file__).resolve().parents[1]


def native_answer_functions(dataset):
    path = ROOT / f"reproduction/provenance/official_evaluators/{dataset}_official_evaluate.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    wanted = {"normalize_answer", "f1_score", "exact_match_score"}
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    if {node.name for node in definitions} != wanted:
        raise ValueError("Pinned native metric definitions are missing")
    # Execute the three unmodified function ASTs with their standard-library
    # dependencies. The original script's unrelated ujson/I/O entry point is
    # not imported, and its answer normalization/scoring bodies are unchanged.
    namespace = {"re": re, "string": string, "Counter": Counter}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["exact_match_score"], namespace["f1_score"]


def alias_mapping(path):
    result = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            entry = json.loads(line)
            key = entry["Q_id"]
            if key in result:
                raise ValueError("Duplicate alias entity ID")
            result[key] = entry.get("aliases", []) + entry.get("demonyms", [])
    return result


def evaluate(filepath_with_predictions, filepath_with_ground_truths, qid=None):
    benchmark = json.loads(Path(os.environ.get("SHGRAPH_BENCHMARK_CONFIG", ROOT / "reproduction/benchmark.json")).read_text(encoding="utf-8"))
    dataset = benchmark["id"]
    exact_match, native_f1 = native_answer_functions(dataset)
    predictions = json.loads(Path(filepath_with_predictions).read_text(encoding="utf-8"))
    native = [json.loads(line) for line in Path(filepath_with_ground_truths).read_text(
        encoding="utf-8"
    ).splitlines() if line.strip()]
    gold = {row.get("id") or row["_id"]: row for row in native}
    if len(gold) != len(native):
        raise ValueError("Duplicate native gold ID")
    ids = list(qid) if qid is not None else list(gold)
    if not ids or len(set(ids)) != len(ids) or set(ids) - set(gold):
        raise ValueError("Frozen IDs must be unique and present in native gold")
    answers = predictions["answer"]
    if set(answers) - set(ids):
        raise ValueError("Predictions outside the frozen sample")
    aliases = alias_mapping(ROOT / benchmark["aliases"]) if benchmark.get("aliases") else {}
    per_question = []
    for identifier in ids:
        row = gold[identifier]
        answer = answers.get(identifier, "")
        if not isinstance(answer, str):
            raise ValueError(f"Non-string prediction: {identifier}")
        candidates = [row["answer"]]
        entity_ids = row.get("answer_id", [])
        if isinstance(entity_ids, str):
            entity_ids = [entity_ids]
        for entity_id in entity_ids or []:
            candidates.extend(aliases.get(entity_id, []))
        present = identifier in answers
        per_question.append({
            "id": identifier, "prediction_present": present,
            "answer_em": float(max(exact_match(answer, label) for label in candidates)) if present else 0.0,
            "answer_f1": max(native_f1(answer, label)[0] for label in candidates) if present else 0.0,
            "canonical_answer_em": float(exact_match(answer, row["answer"])) if present else 0.0,
            "canonical_answer_f1": native_f1(answer, row["answer"])[0] if present else 0.0,
            "question_type": row.get("type", ""),
            "gold_alias_count": len(candidates) - 1,
        })
    em = sum(row["answer_em"] for row in per_question) / len(ids)
    f1 = sum(row["answer_f1"] for row in per_question) / len(ids)
    return {
        "dataset": dataset, "answer_em": em, "answer_f1": f1,
        "answer_em_percent": 100 * em, "answer_f1_percent": 100 * f1,
        "canonical_answer_em": sum(row["canonical_answer_em"] for row in per_question) / len(ids),
        "canonical_answer_f1": sum(row["canonical_answer_f1"] for row in per_question) / len(ids),
        "denominator": len(ids), "prediction_count": len(answers),
        "prediction_coverage": len(answers) / len(ids),
        "missing_prediction_ids": [identifier for identifier in ids if identifier not in answers],
        "per_question": per_question,
        "metric_definition": "Native unmodified EM/F1 functions; 2Wiki v1.1 aliases and demonyms; fixed frozen IDs",
        "supporting_fact_metrics": None,
        "supporting_fact_metrics_note": "Supplied QA returns chunk references rather than native sentence-level predictions",
    }
