"""Verify the upstream corpus boundary from pinned code and local data.

This offline audit never constructs a graph or sends model requests. Gold
support labels are counted for documentation only, after the sample is frozen.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

import pyarrow.parquet as pq

from reproduction.campaign import DEFAULT_RUN
from reproduction.freeze_musique_sample import DESTINATION, ROOT, sha256
from reproduction.runtime import utc_now, write_json

UPSTREAM = "3ed1685fefb1295c43c7fca0bb306e4d5fd08e77"


def passage_texts(row):
    return [
        title + "\n" + "\n".join(sentences)
        for title, sentences in zip(
            row["context"]["title"], row["context"]["sentences"], strict=True
        )
    ]


def describe(rows):
    texts = [text for row in rows for text in passage_texts(row)]
    counts = Counter(texts)
    return {
        "questions": len(rows),
        "candidate_occurrences": len(texts),
        "exact_unique_passages": len(counts),
        "repeated_occurrences": len(texts) - len(counts),
        "candidate_text_utf8_bytes_with_occurrences": sum(
            len(text.encode("utf-8")) for text in texts
        ),
        "exact_unique_candidate_text_utf8_bytes": sum(
            len(text.encode("utf-8")) for text in counts
        ),
    }, set(counts)


def upstream_file(name):
    return subprocess.check_output(
        ["git", "show", f"{UPSTREAM}:{name}"], cwd=ROOT
    )


def matching_lines(content, text):
    return [
        {"line": number, "text": line}
        for number, line in enumerate(content.decode("utf-8").splitlines(), 1)
        if text in line
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    manifest = json.loads(
        (DESTINATION / "manifest.json").read_text(encoding="utf-8")
    )
    source_path = Path(manifest["full_development_parquet"]["path"])
    if sha256(source_path) != manifest["full_development_parquet"]["sha256"]:
        raise ValueError("Full input source no longer matches the frozen manifest")
    all_rows = pq.read_table(source_path).to_pylist()
    sample_rows = pq.read_table(DESTINATION / "questions.parquet").to_pylist()
    ids = json.loads((DESTINATION / "ids.json").read_text(encoding="utf-8"))
    if [row["id"] for row in sample_rows] != ids:
        raise ValueError("Sample IDs or order differ from the frozen IDs")

    full, full_texts = describe(all_rows)
    first500, _ = describe(all_rows[:500])
    sample, sample_texts = describe(sample_rows)
    if (sample["candidate_occurrences"], sample["exact_unique_passages"]) != (
        manifest["candidate_occurrences"], manifest["unique_candidate_passages"]
    ):
        raise ValueError("Sample context statistics differ from the manifest")
    exported_texts = {
        json.loads(line)["text"]
        for line in (DESTINATION / "candidate_corpus.jsonl")
        .read_text(encoding="utf-8").splitlines()
    }
    if sample_texts != exported_texts:
        raise ValueError("Candidate export differs from all sampled contexts")

    native = [
        json.loads(line)
        for line in (DESTINATION / "gold.jsonl")
        .read_text(encoding="utf-8").splitlines()
    ]
    if [row["id"] for row in native] != ids:
        raise ValueError("Native audit labels are not aligned to the sample")
    support = Counter()
    for row, original in zip(sample_rows, native, strict=True):
        expected = [
            paragraph["title"] + "\n" + paragraph["paragraph_text"]
            for paragraph in original["paragraphs"]
        ]
        if passage_texts(row) != expected:
            raise ValueError(f"Candidate content/order differs: {row['id']}")
        for paragraph in original["paragraphs"]:
            if not isinstance(paragraph.get("is_supporting"), bool):
                raise ValueError("Missing native Boolean support label")
            support[
                "gold_supporting_occurrences"
                if paragraph["is_supporting"] else "distractor_occurrences"
            ] += 1
    if sum(support.values()) != sample["candidate_occurrences"]:
        raise ValueError("Support/distractor labels do not cover every candidate")

    processor = upstream_file("processor/dataset_processor.py")
    builder = upstream_file("build/build_graph.py")
    selector = matching_lines(processor, ".select(range(500))")
    context_loop = matching_lines(processor, 'zip(data["context"]["title"]')
    process_call = matching_lines(builder, "self.dataset_processor.process_dataset()")
    if not selector or not context_loop or not process_call:
        raise ValueError("Pinned upstream corpus-boundary evidence changed")
    report = {
        "generated_at": utc_now(),
        "audit_is_offline": True,
        "new_model_api_requests": 0,
        "upstream_commit": UPSTREAM,
        "upstream_processor_sha256": hashlib.sha256(processor).hexdigest(),
        "upstream_builder_sha256": hashlib.sha256(builder).hexdigest(),
        "source_evidence": {
            "default_question_selection": selector,
            "candidate_context_iteration": context_loop,
            "builder_processor_handoff": process_call,
        },
        "full_development_contexts_statistics_only": full,
        "upstream_default_first500_contexts_statistics_only": first500,
        "executed_seed42_sample_contexts": {**sample, **dict(support)},
        "full_development_unique_passages_absent_from_sample": len(
            full_texts - sample_texts
        ),
        "sample_unique_passages_subset_of_development": sample_texts <= full_texts,
        "checks_passed": {
            "frozen_ids_and_order": True,
            "all_sample_candidates_equal_native_contexts": True,
            "distractors_included_in_build_input": support["distractor_occurrences"] > 0,
            "candidate_export_equals_all_sample_contexts": True,
        },
        "interpretation": [
            "Pinned upstream selects first500 questions and iterates every candidate context",
            "Executed change substitutes seed42 n1000 sampling for first500; candidate scope is retained",
            "Support labels are counted only in this offline audit, not used to select build passages",
            "Neither all development contexts nor full Wikipedia are built by this campaign",
            "Original question-specific entity scopes also require matching IDs when reusing a graph",
            "Exact repeated passage extraction is reusable through the original cache",
        ],
    }
    write_json(args.run_dir / "corpus_scope_audit.json", report)
    print(json.dumps({
        "sample": report["executed_seed42_sample_contexts"],
        "full_development_contexts_statistics_only": full,
        "checks_passed": report["checks_passed"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
