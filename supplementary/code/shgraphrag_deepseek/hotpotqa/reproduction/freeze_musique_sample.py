"""Freeze and validate the approved MuSiQue sample using local files only.

Run with the isolated environment:
    .venv/Scripts/python.exe -m reproduction.freeze_musique_sample
    .venv/Scripts/python.exe -m reproduction.freeze_musique_sample --validate-only

The upstream DATASET_DIR receives an exact copy of the full supplied development
Parquet. DatasetProcessor samples that full file, using the same selector here.
The separate 1000-row Parquet is an ID-aligned inference/evaluation handoff;
feeding it back into DatasetProcessor would apply sampling twice.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys

import pyarrow as pa
import pyarrow.parquet as pq

from config.settings import DATASET_DIR, SAMPLE_SEED, SAMPLE_SIZE
from processor.sample_selection import sample_indices

ROOT = Path(__file__).resolve().parents[1]
SUPPLIED = ROOT.parent / "SHGraphRAG"
DEFAULT_SOURCE = SUPPLIED / "dataset/musique/musique_ans_dev.parquet"
DEFAULT_GOLD = (
    SUPPLIED / "dataset/raw_official/musique/release_v1/data/musique_ans_v1.0_dev.jsonl"
)
DESTINATION = ROOT / "reproduction/artifacts/musique_seed42_n1000"
EXPECTED_GOLD_SHA256 = (
    "15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b"
)
ALGORITHM = "Python random.Random(seed).sample(sorted unique question IDs, sample_size)"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def read_sources(source: Path, native_gold: Path):
    require(sha256(native_gold) == EXPECTED_GOLD_SHA256, "Unexpected native gold release")
    table = pq.read_table(source)
    rows = table.to_pylist()
    gold = read_jsonl(native_gold)
    require(len(rows) == len(gold) == 2417, "Expected 2,417 MuSiQue-Ans dev questions")
    by_id = {row["id"]: row for row in gold}
    require(len(by_id) == len(gold), "Native gold contains duplicate IDs")
    require(set(row["id"] for row in rows) == set(by_id), "Parquet/gold ID sets differ")
    for row in rows:
        original = by_id[row["id"]]
        for field in ("question", "answer", "answerable"):
            require(row[field] == original[field], f"Source mismatch: {row['id']}/{field}")
        require(
            row["context"]["title"] == [p["title"] for p in original["paragraphs"]],
            f"Candidate title/order mismatch: {row['id']}",
        )
        require(
            row["context"]["sentences"]
            == [[p["paragraph_text"]] for p in original["paragraphs"]],
            f"Candidate text/order mismatch: {row['id']}",
        )
    return table, rows, by_id


def corpus_artifacts(rows: list[dict]):
    corpus: dict[str, dict] = {}
    mappings = []
    for row in rows:
        passage_ids = []
        for title, sentences in zip(
            row["context"]["title"], row["context"]["sentences"], strict=True
        ):
            # Match DatasetProcessor.process_data exactly, including newlines.
            text = title + "\n" + "\n".join(sentences)
            pid = hashlib.sha256(text.encode("utf-8")).hexdigest()
            passage_ids.append(pid)
            passage = {"id": pid, "title": title, "sentences": sentences, "text": text}
            require(pid not in corpus or corpus[pid] == passage, "Passage hash collision")
            corpus[pid] = passage
        mappings.append({"question_id": row["id"], "passage_ids": passage_ids})
    return list(corpus.values()), mappings


def freeze(source: Path, native_gold: Path, destination: Path) -> dict:
    table, rows, gold_by_id = read_sources(source, native_gold)
    indices = sample_indices([row["id"] for row in rows], SAMPLE_SIZE, SAMPLE_SEED)
    sample_rows = [rows[index] for index in indices]
    ids = [row["id"] for row in sample_rows]
    corpus, mappings = corpus_artifacts(sample_rows)
    if DATASET_DIR.exists():
        require(sha256(DATASET_DIR) == sha256(source), "Existing DATASET_DIR differs")
    else:
        DATASET_DIR.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, DATASET_DIR)
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "ids.json", ids)
    write_json(destination / "source_indices.json", indices)
    pq.write_table(table.take(pa.array(indices, type=pa.int64())), destination / "questions.parquet")
    write_jsonl(destination / "gold.jsonl", [gold_by_id[qid] for qid in ids])
    write_jsonl(destination / "candidate_corpus.jsonl", corpus)
    write_jsonl(destination / "question_passages.jsonl", mappings)
    files = {
        name: {"sha256": sha256(destination / name), "bytes": (destination / name).stat().st_size}
        for name in (
            "ids.json", "source_indices.json", "questions.parquet", "gold.jsonl",
            "candidate_corpus.jsonl", "question_passages.jsonl",
        )
    }
    occurrences = sum(len(row["passage_ids"]) for row in mappings)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "MuSiQue-Ans v1.0",
        "split": "development",
        "status": "new_user_approved_sample_not_original_paper_sample",
        "upstream_commit": "3ed1685fefb1295c43c7fca0bb306e4d5fd08e77",
        "seed": SAMPLE_SEED,
        "sample_size": SAMPLE_SIZE,
        "source_rows": len(rows),
        "sampling": ALGORITHM,
        "order": "RNG draw order, shared by construction, inference, and evaluation",
        "source_parquet": {"path": str(source), "sha256": sha256(source)},
        "native_gold_source": {"path": str(native_gold), "sha256": sha256(native_gold)},
        "full_development_parquet": {
            "path": str(DATASET_DIR), "sha256": sha256(DATASET_DIR),
            "role": "Upstream DATASET_DIR; sampled once by DatasetProcessor",
        },
        "scope": "Only the official candidate paragraphs belonging to the sampled questions",
        "candidate_occurrences": occurrences,
        "unique_candidate_passages": len(corpus),
        "candidate_deduplication": "Statistics/export only; original build keeps question-specific occurrences",
        "hop_counts": dict(Counter(qid.split("__", 1)[0] for qid in ids)),
        "answerable_counts": dict(Counter(str(row["answerable"]) for row in sample_rows)),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in ("datasets", "numpy", "pyarrow")},
        "selection_source_sha256": sha256(ROOT / "processor/sample_selection.py"),
        "preparation_script_sha256": sha256(Path(__file__)),
        "files": files,
        "llm_calls_for_preparation": 0,
    }
    write_json(destination / "manifest.json", manifest)
    return manifest


def validate(source: Path, native_gold: Path, destination: Path) -> dict:
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    table, rows, gold_by_id = read_sources(source, native_gold)
    require(manifest["seed"] == SAMPLE_SEED == 42, "Sample seed differs")
    require(manifest["sample_size"] == SAMPLE_SIZE == 1000, "Sample size differs")
    require(sha256(source) == manifest["source_parquet"]["sha256"], "Source hash differs")
    require(sha256(DATASET_DIR) == manifest["full_development_parquet"]["sha256"], "Build input differs")
    require(sha256(source) == sha256(DATASET_DIR), "Full build input is not a source copy")
    for name, record in manifest["files"].items():
        require(sha256(destination / name) == record["sha256"], f"Artifact hash differs: {name}")
    require(sha256(ROOT / "processor/sample_selection.py") == manifest["selection_source_sha256"], "Selector code changed")
    require(sha256(Path(__file__)) == manifest["preparation_script_sha256"], "Preparation code changed")
    indices = sample_indices([row["id"] for row in rows], SAMPLE_SIZE, SAMPLE_SEED)
    expected = [rows[index] for index in indices]
    ids = json.loads((destination / "ids.json").read_text(encoding="utf-8"))
    require(ids == [row["id"] for row in expected], "Frozen IDs/order do not match seed")
    require(len(set(ids)) == SAMPLE_SIZE, "Sample IDs are not unique")
    require(
        json.loads((destination / "source_indices.json").read_text(encoding="utf-8")) == indices,
        "Source indices/order do not match seed",
    )
    require(pq.read_table(destination / "questions.parquet").to_pylist() == expected, "Sample Parquet differs")
    require(read_jsonl(destination / "gold.jsonl") == [gold_by_id[qid] for qid in ids], "Native gold differs")
    corpus, mappings = corpus_artifacts(expected)
    require(read_jsonl(destination / "candidate_corpus.jsonl") == corpus, "Candidate corpus differs")
    require(read_jsonl(destination / "question_passages.jsonl") == mappings, "Passage membership/order differs")
    report = {
        "passed": True,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "checks": [
            "Complete source matches official gold, questions, answers, candidates, and their order",
            "Full upstream build input is a byte-identical source copy",
            "Seed 42, 1000 unique IDs, and draw order reproduced",
            "Sample Parquet and native gold aligned exactly by ID",
            "Candidate text, corpus, and per-question membership reproduced",
            "All artifact and preparation-code hashes match",
        ],
        "sample_size": len(ids),
        "source_rows": len(table),
        "candidate_occurrences": sum(len(row["passage_ids"]) for row in mappings),
        "unique_candidate_passages": len(corpus),
        "llm_calls": 0,
    }
    write_json(destination / "validation.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-parquet", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--native-gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--output-dir", type=Path, default=DESTINATION)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if not args.validate_only and not args.output_dir.exists():
        freeze(args.source_parquet.resolve(), args.native_gold.resolve(), args.output_dir.resolve())
    result = validate(args.source_parquet.resolve(), args.native_gold.resolve(), args.output_dir.resolve())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
