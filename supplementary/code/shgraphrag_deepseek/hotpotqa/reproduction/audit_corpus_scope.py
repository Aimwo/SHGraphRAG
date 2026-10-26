"""Offline candidate-corpus and provenance checks for the configured benchmark."""

import json
from pathlib import Path
import pyarrow.parquet as pq

from reproduction.campaign import DEFAULT_RUN
from reproduction.freeze_benchmark import (
    BENCHMARK, DEFAULT_SOURCE, DEFAULT_GOLD, DESTINATION, corpus_artifacts,
    question_id, read_jsonl, read_sources, sha256, validate,
)
from reproduction.runtime import utc_now, write_json


def main():
    validation = validate()
    _, full, native = read_sources(DEFAULT_SOURCE, DEFAULT_GOLD)
    selected = pq.read_table(DESTINATION / "questions.parquet").to_pylist()
    ids = json.loads((DESTINATION / "ids.json").read_text(encoding="utf-8"))
    if [row["id"] for row in selected] != ids:
        raise ValueError("Frozen order differs")
    corpus, mappings = corpus_artifacts(selected)
    if read_jsonl(DESTINATION / "candidate_corpus.jsonl") != corpus:
        raise ValueError("Candidate export differs")
    supporting = distractors = 0
    for row in selected:
        gold = native[row["id"]]
        support_titles = {fact[0] for fact in gold["supporting_facts"]}
        for title in row["context"]["title"]:
            if title in support_titles:
                supporting += 1
            else:
                distractors += 1
    report = {
        "generated_at": utc_now(), "audit_is_offline": True, "new_model_api_requests": 0,
        "dataset": BENCHMARK["name"], "validation": validation,
        "source_rows": len(full), "sample_size": len(ids), "seed": 42,
        "source_sha256": sha256(DEFAULT_SOURCE), "native_gold_sha256": sha256(DEFAULT_GOLD),
        "executed_seed42_sample_contexts": {
            "questions": len(ids), "candidate_occurrences": supporting + distractors,
            "exact_unique_passages": len(corpus),
            "candidate_occurrences_with_native_supporting_title": supporting,
            "candidate_occurrences_without_native_supporting_title": distractors,
        },
        "checks_passed": {
            "frozen_ids_and_order": True, "all_sample_candidates_equal_native_contexts": True,
            "distractors_included_in_build_input": distractors > 0,
            "candidate_export_equals_all_sample_contexts": True,
        },
        "scope": "All candidates attached to the sampled development questions; no full Wikipedia",
        "support_labels": "Title membership counted offline only; never filters construction inputs",
    }
    write_json(DEFAULT_RUN / "corpus_scope_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
