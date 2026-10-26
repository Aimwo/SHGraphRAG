"""Freeze the configured native benchmark with the approved seed42 selector."""

import argparse
import os
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

import pyarrow as pa
import pyarrow.parquet as pq

from config.settings import DATASET_DIR, SAMPLE_SEED, SAMPLE_SIZE
from processor.sample_selection import sample_indices

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = Path(os.environ.get("SHGRAPH_BENCHMARK_CONFIG",
    ROOT / "reproduction/benchmark.json"))
BENCHMARK = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
DEFAULT_SOURCE = ROOT / BENCHMARK["source"]
DEFAULT_GOLD = ROOT / BENCHMARK["native_gold"]
DESTINATION = ROOT / f"reproduction/artifacts/{BENCHMARK['id']}_seed42_n1000"
ALGORITHM = "Python random.Random(seed).sample(sorted unique question IDs, sample_size)"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]


def question_id(row):
    return row.get("id") or row["_id"]


def read_native(path):
    if BENCHMARK["native_gold_format"] == "json":
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return read_jsonl(path)


def read_sources(source, native_gold):
    require(sha256(source) == BENCHMARK["source_sha256"], "Full normalized source hash differs")
    require(sha256(native_gold) == BENCHMARK["native_gold_sha256"], "Native source hash differs")
    if BENCHMARK.get("aliases"):
        require(sha256(ROOT / BENCHMARK["aliases"]) == BENCHMARK["aliases_sha256"],
                "Native 2Wiki alias/demonym release hash differs")
    table = pq.read_table(source)
    rows, native = table.to_pylist(), read_native(native_gold)
    require(len(rows) == len(native) == BENCHMARK["source_rows"], "Wrong development split size")
    by_id = {question_id(row): row for row in native}
    require(len(by_id) == len(native), "Duplicate native question IDs")
    require(set(row["id"] for row in rows) == set(by_id), "Full source/gold ID mismatch")
    for row in rows:
        original = by_id[row["id"]]
        require(row["question"] == original["question"] and row["answer"] == original["answer"],
                f"Question or answer mismatch: {row['id']}")
        expected = {
            "title": [entry[0] for entry in original["context"]],
            "sentences": [entry[1] for entry in original["context"]],
        }
        require(row["context"] == expected, f"Native candidate content/order differs: {row['id']}")
    return table, rows, by_id


def corpus_artifacts(rows):
    corpus, mappings = {}, []
    for row in rows:
        passage_ids = []
        for title, sentences in zip(row["context"]["title"], row["context"]["sentences"], strict=True):
            text = title + "\n" + "\n".join(sentences)
            identifier = hashlib.sha256(text.encode("utf-8")).hexdigest()
            passage = {"id": identifier, "title": title, "sentences": sentences, "text": text}
            require(identifier not in corpus or corpus[identifier] == passage, "Passage hash collision")
            corpus[identifier] = passage
            passage_ids.append(identifier)
        mappings.append({"question_id": row["id"], "passage_ids": passage_ids})
    return list(corpus.values()), mappings


def freeze(source=DEFAULT_SOURCE, native_gold=DEFAULT_GOLD, destination=DESTINATION):
    require(SAMPLE_SIZE == 1000 and SAMPLE_SEED == 42, "Approved sample is seed42 n1000")
    table, rows, gold = read_sources(source, native_gold)
    indices = sample_indices([row["id"] for row in rows], SAMPLE_SIZE, SAMPLE_SEED)
    selected = [rows[index] for index in indices]
    ids = [row["id"] for row in selected]
    corpus, mappings = corpus_artifacts(selected)
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "ids.json", ids)
    write_json(destination / "source_indices.json", indices)
    pq.write_table(table.take(pa.array(indices, type=pa.int64())), destination / "questions.parquet")
    write_jsonl(destination / "gold.jsonl", [gold[qid] for qid in ids])
    write_jsonl(destination / "candidate_corpus.jsonl", corpus)
    write_jsonl(destination / "question_passages.jsonl", mappings)
    files = {
        name: {"sha256": sha256(destination / name), "bytes": (destination / name).stat().st_size}
        for name in ("ids.json", "source_indices.json", "questions.parquet", "gold.jsonl",
                     "candidate_corpus.jsonl", "question_passages.jsonl")
    }
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": BENCHMARK["name"], "dataset_id": BENCHMARK["id"], "split": "development",
        "status": "new_user_approved_sample_not_original_paper_sample",
        "upstream_commit": "3ed1685fefb1295c43c7fca0bb306e4d5fd08e77",
        "seed": SAMPLE_SEED, "sample_size": SAMPLE_SIZE, "source_rows": len(rows),
        "sampling": ALGORITHM, "order": "RNG draw order shared by construction, QA and evaluation",
        "source_parquet": {"path": str(source), "sha256": sha256(source)},
        "native_gold_source": {"path": str(native_gold), "sha256": sha256(native_gold)},
        "full_development_parquet": {
            "path": str(DATASET_DIR), "sha256": sha256(DATASET_DIR),
            "role": "Full source sampled exactly once by DatasetProcessor",
        },
        "scope": "All original candidate paragraphs of the frozen development questions",
        "candidate_occurrences": sum(len(row["passage_ids"]) for row in mappings),
        "unique_candidate_passages": len(corpus),
        "candidate_deduplication": "Exact extraction reuse; question memberships retained",
        "question_types": dict(Counter(row.get("type", "") for row in selected)),
        "python": sys.version, "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in ("datasets", "numpy", "pyarrow")},
        "selection_source_sha256": sha256(ROOT / "processor/sample_selection.py"),
        "preparation_script_sha256": sha256(Path(__file__)),
        "benchmark_config_sha256": sha256(BENCHMARK_PATH),
        "files": files, "llm_calls_for_preparation": 0,
    }
    write_json(destination / "manifest.json", manifest)
    return manifest


def validate(source=DEFAULT_SOURCE, native_gold=DEFAULT_GOLD, destination=DESTINATION):
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    table, rows, gold = read_sources(source, native_gold)
    require(manifest["seed"] == SAMPLE_SEED == 42, "Wrong seed")
    require(manifest["sample_size"] == SAMPLE_SIZE == 1000, "Wrong sample size")
    require(sha256(source) == sha256(DATASET_DIR) == manifest["source_parquet"]["sha256"],
            "Build source is not the pinned full development file")
    for name, info in manifest["files"].items():
        require(sha256(destination / name) == info["sha256"], f"Frozen artifact changed: {name}")
    for path, field in (
        (ROOT / "processor/sample_selection.py", "selection_source_sha256"),
        (Path(__file__), "preparation_script_sha256"),
        (BENCHMARK_PATH, "benchmark_config_sha256"),
    ):
        require(sha256(path) == manifest[field], f"Preparation source changed: {path.name}")
    indices = sample_indices([row["id"] for row in rows], SAMPLE_SIZE, SAMPLE_SEED)
    expected = [rows[index] for index in indices]
    ids = json.loads((destination / "ids.json").read_text(encoding="utf-8"))
    require(ids == [row["id"] for row in expected] and len(set(ids)) == 1000, "Wrong ID/order")
    require(json.loads((destination / "source_indices.json").read_text(encoding="utf-8")) == indices,
            "Frozen original row order differs")
    require(pq.read_table(destination / "questions.parquet").to_pylist() == expected,
            "QA handoff differs from the selected full-source rows")
    require(read_jsonl(destination / "gold.jsonl") == [gold[qid] for qid in ids], "Native gold differs")
    corpus, mappings = corpus_artifacts(expected)
    require(read_jsonl(destination / "candidate_corpus.jsonl") == corpus, "Candidate export differs")
    require(read_jsonl(destination / "question_passages.jsonl") == mappings, "Memberships differ")
    report = {
        "passed": True, "checked_at": datetime.now(timezone.utc).isoformat(),
        "dataset": BENCHMARK["name"], "sample_size": 1000, "source_rows": len(table),
        "candidate_occurrences": manifest["candidate_occurrences"],
        "unique_candidate_passages": manifest["unique_candidate_passages"],
        "checks": ["Exact source/native hashes", "Seed42 frozen draw order", "All candidate texts/order",
                   "1000 unique IDs and native gold", "All artifact and preparation-code hashes"],
        "llm_calls": 0,
    }
    write_json(destination / "validation.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    result = validate() if args.validate_only else freeze()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
