"""Reconstruct both published protocols from reader-supplied official dev files.

No downloads, model calls, credential reads, or modifications to the input files.
The output contains data and must remain outside version control.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import shutil

SUPPLEMENT = Path(__file__).resolve().parents[1]
RAW = {
    "hotpotqa": ("dataset/hotpot_qa/distractor/validation-00000-of-00001.parquet",
                "c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6"),
    "2wiki": ("dataset/raw_official/2wiki/release_20210407/dev.json",
              "79f77ae104088ea8e25b1a65dbece768d45771194663bc5660ec9a98070dadf5"),
    "musique": ("dataset/raw_official/musique/release_v1/data/musique_ans_v1.0_dev.jsonl",
                "15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b"),
}
ALIASES = ("dataset/raw_official/2wiki/release_20210407/id_aliases.json",
           "f08ffcb6c2cefca9bdbe86b4248d6ad7a7743762d3f7264c14ff0bae85726fb6")
COUNTS = {"hotpotqa": 7405, "2wiki": 12576, "musique": 2417}
PASSAGES = {
    20261020: {"hotpotqa": 9800, "2wiki": 6293, "musique": 11524},
    42: {"hotpotqa": 9773, "2wiki": 6361, "musique": 11835},
}


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, ensure_ascii=False) + "\n").encode("utf-8"))


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def record(path, root):
    return {"path": Path(path).relative_to(root).as_posix(),
            "sha256": sha256(path), "bytes": Path(path).stat().st_size}


def load_converters():
    spec = importlib.util.spec_from_file_location(
        "original_preparation", SUPPLEMENT / "tools/original_prepare_paper_datasets.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_official(source_root, output_root):
    import pyarrow as pa
    import pyarrow.parquet as pq

    converter = load_converters()
    tables, gold, sources = {}, {}, {}
    for dataset, (relative, expected) in RAW.items():
        path = source_root / relative
        if sha256(path) != expected:
            raise ValueError(f"Wrong official release: {relative}")
        sources[dataset] = record(path, source_root)
        if dataset == "hotpotqa":
            table = pq.read_table(path)
            native = [converter.hotpot_native(row) for row in table.to_pylist()]
        elif dataset == "2wiki":
            native = read_json(path)
            table = pa.Table.from_pylist([converter.two_wiki_parquet_row(r) for r in native])
        else:
            native = [json.loads(line) for line in path.read_text(
                encoding="utf-8-sig").splitlines() if line.strip()]
            table = pa.Table.from_pylist([converter.musique_parquet_row(r) for r in native])
        rows = table.to_pylist()
        by_id = {r["id"]: r for r in rows}
        gold_by_id = {r.get("_id") or r["id"]: r for r in native}
        if len(rows) != COUNTS[dataset] or len(by_id) != len(rows) or set(by_id) != set(gold_by_id):
            raise ValueError(f"Duplicate IDs, wrong split size, or native alignment: {dataset}")
        if not all(r["question"] and r["answer"] for r in rows):
            raise ValueError("Unexpected empty question/answer in source release")
        tables[dataset], gold[dataset] = table, gold_by_id
        normal = output_root / "normalized" / f"{dataset}.parquet"
        normal.parent.mkdir(parents=True, exist_ok=True)
        if dataset == "hotpotqa":
            if path.resolve() != normal.resolve():
                shutil.copyfile(path, normal)
        else:
            pq.write_table(table, normal)
        # The native MuSiQue freezer and the GPT alias scorer use these paths.
        if dataset == "musique":
            target = output_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if path.resolve() != target.resolve():
                shutil.copyfile(path, target)
    path = source_root / ALIASES[0]
    if sha256(path) != ALIASES[1]:
        raise ValueError("Wrong 2Wiki official alias/demonym release")
    target = output_root / ALIASES[0]
    target.parent.mkdir(parents=True, exist_ok=True)
    if path.resolve() != target.resolve():
        shutil.copyfile(path, target)
    sources["2wiki_aliases"] = record(path, source_root)
    return tables, gold, sources, converter


def select_ids(dataset, rows, seed):
    by_id = {r["id"]: r for r in rows}
    ids = random.Random(seed).sample(sorted(by_id), 1000)
    expected = read_json(SUPPLEMENT / "protocols" / f"seed{seed}" / dataset / "ids.json")
    if ids != expected:
        raise ValueError(f"Sample ID/order differs from accepted campaign: {dataset}, {seed}")
    return ids


def prepare_gpt(root, tables, gold, sources, converter):
    import pyarrow.parquet as pq

    converter.ROOT = root
    manifest = {
        "prepared_on": datetime.now(timezone.utc).date().isoformat(), "sample_seed": 20261020,
        "sampling": "Python random.Random(seed).sample(sorted unique IDs, 1000)",
        "sample_status": "new_revision_samples_not_original_Gemini_paper_samples",
        "official_sources": sources, "datasets": {},
    }
    lock = {"schema_version": 1, "scope": "union of sample candidate passages",
            "full_corpus_experiment": False, "datasets": []}
    historical = read_json(SUPPLEMENT / "protocols/seed20261020/original_manifest.json")["content"]
    for dataset, table in tables.items():
        rows = table.to_pylist()
        by_id = {r["id"]: r for r in rows}
        ids = select_ids(dataset, rows, 20261020)
        folder = root / "dataset/paper_ready" / dataset / "sample_1000_seed20261020"
        folder.mkdir(parents=True, exist_ok=True)
        write_json(folder / "ids.json", ids)
        native = [gold[dataset][qid] for qid in ids]
        gold_file = folder / ("gold.jsonl" if dataset == "musique" else "gold.json")
        (write_jsonl if dataset == "musique" else write_json)(gold_file, native)
        indices = {r["id"]: i for i, r in enumerate(rows)}
        pq.write_table(table.take([indices[qid] for qid in ids]), folder / "questions.parquet")
        corpus = converter.candidate_corpus([by_id[qid] for qid in ids], folder)
        if corpus["unique_passages"] != PASSAGES[20261020][dataset]:
            raise ValueError("Candidate passage count differs")
        original = historical["datasets"][dataset]
        for key in ("corpus_sha256", "question_passages_sha256"):
            if corpus[key] != original["candidate_corpus"][key]:
                raise ValueError(f"Historical corpus bytes differ: {dataset}/{key}")
        for name, expected in (("ids.json", original["sample_ids_sha256"]),
                               (gold_file.name, original["sample_gold_sha256"])):
            if sha256(folder / name) != expected:
                raise ValueError(f"Historical text artifact bytes differ: {dataset}/{name}")
        queries = root / "gpt4omini/queries" / f"{dataset}.jsonl"
        write_jsonl(queries, ({"id": qid, "question": by_id[qid]["question"]} for qid in ids))
        info = {
            "dev_rows": len(rows), "sample_size": 1000, "candidate_corpus": corpus,
            "sample_ids": (folder / "ids.json").relative_to(root).as_posix(),
            "sample_ids_sha256": sha256(folder / "ids.json"),
            "sample_gold": gold_file.relative_to(root).as_posix(),
            "sample_gold_sha256": sha256(gold_file),
            "sample_parquet": (folder / "questions.parquet").relative_to(root).as_posix(),
            "sample_parquet_sha256": sha256(folder / "questions.parquet"),
        }
        if dataset == "2wiki":
            info.update(answer_aliases=ALIASES[0], answer_aliases_sha256=ALIASES[1])
        manifest["datasets"][dataset] = info
        lock["datasets"].append({
            "dataset": dataset, "sample_seed": 20261020, "question_count": 1000,
            "candidate_passage_count": corpus["unique_passages"],
            "inference_queries": record(queries, root),
            "sources": {
                "ids": record(folder / "ids.json", root),
                "native_gold_evaluation_only": record(gold_file, root),
                "candidate_corpus": record(folder / "candidate_corpus.jsonl", root),
                "question_passages_evaluation_only": record(folder / "question_passages.jsonl", root),
                "normalized_questions_evaluation_only": record(folder / "questions.parquet", root),
            },
        })
    write_json(root / "dataset/paper_ready/manifest.json", manifest)
    write_json(root / "gpt4omini/datasets.lock.json", lock)


def prepare_deepseek(root, tables, gold):
    for dataset, table in tables.items():
        rows = table.to_pylist()
        by_id = {r["id"]: r for r in rows}
        ids = select_ids(dataset, rows, 42)
        folder = root / "deepseek" / dataset
        folder.mkdir(parents=True, exist_ok=True)
        write_json(folder / "ids.json", ids)
        write_jsonl(folder / "queries.jsonl",
                    ({"id": qid, "question": by_id[qid]["question"]} for qid in ids))
        write_jsonl(folder / "gold.jsonl", (gold[dataset][qid] for qid in ids))
        passages, mappings = {}, []
        for qid in ids:
            row, passage_ids = by_id[qid], []
            for title, sentences in zip(row["context"]["title"], row["context"]["sentences"], strict=True):
                text = title + "\n" + "\n".join(sentences)
                pid = hashlib.sha256(text.encode("utf-8")).hexdigest()
                item = {"id": pid, "title": title, "sentences": sentences, "text": text}
                if pid in passages and passages[pid] != item:
                    raise ValueError("Passage ID collision")
                passages[pid] = item
                passage_ids.append(pid)
            mappings.append({"question_id": qid, "passage_ids": passage_ids})
        if len(passages) != PASSAGES[42][dataset]:
            raise ValueError("DeepSeek candidate passage count differs")
        write_jsonl(folder / "candidate_corpus.jsonl", passages.values())
        write_jsonl(folder / "question_passages.jsonl", mappings)
        alias_rows = [json.loads(line) for line in (root / ALIASES[0]).read_text(
            encoding="utf-8-sig").splitlines() if line.strip()] if dataset == "2wiki" else []
        write_jsonl(folder / "answer_aliases.jsonl", alias_rows)
        names = ("ids.json", "queries.jsonl", "gold.jsonl", "candidate_corpus.jsonl",
                 "question_passages.jsonl", "answer_aliases.jsonl")
        manifest = {
            "dataset_id": dataset, "seed": 42, "count": 1000, "sample_size": 1000,
            "source_rows": len(rows), "unique_candidate_passages": len(passages),
            "candidate_occurrences": sum(len(r["passage_ids"]) for r in mappings),
            "files": {name: {key: value for key, value in record(folder / name, root).items()
                             if key != "path"} for name in names},
        }
        # The byte format follows the original SH freezer for corpus and memberships.
        original = read_json(SUPPLEMENT / "protocols/seed42" / dataset / "original_manifest.json")["content"]
        for name in ("candidate_corpus.jsonl", "question_passages.jsonl", "gold.jsonl"):
            if manifest["files"][name]["sha256"] != original["files"][name]["sha256"]:
                raise ValueError(f"DeepSeek historical input bytes differ: {dataset}/{name}")
        write_json(folder / "manifest.json", manifest)
        if dataset != "musique":
            benchmark = read_json(SUPPLEMENT / "code/shgraphrag_deepseek" / dataset
                                  / "reproduction/benchmark.json")
            full_native = root / "normalized" / f"{dataset}.native.json"
            write_json(full_native, list(gold[dataset].values()))
            normalized = root / "normalized" / f"{dataset}.parquet"
            benchmark.update(
                source=str(normalized.resolve()), source_sha256=sha256(normalized),
                native_gold=str(full_native.resolve()), native_gold_sha256=sha256(full_native),
                native_source={"release_sha256": RAW[dataset][1],
                               "conversion": "Official native fields; original context order retained"},
                aliases=str((root / ALIASES[0]).resolve()) if dataset == "2wiki" else None,
                aliases_sha256=ALIASES[1] if dataset == "2wiki" else None,
            )
            write_json(root / "deepseek/shgraphrag" / dataset / "benchmark.json", benchmark)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True,
                        help="Read-only root containing the four paths in docs/DATA.md")
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Dedicated output directory, excluded from Git")
    args = parser.parse_args()
    source_root, data_root = args.source_root.resolve(), args.data_root.resolve()
    if source_root == data_root or source_root.is_relative_to(data_root):
        parser.error("Use a dedicated output root separate from the read-only source root")
    data_root.mkdir(parents=True, exist_ok=True)
    tables, gold, sources, converter = load_official(source_root, data_root)
    prepare_gpt(data_root, tables, gold, sources, converter)
    prepare_deepseek(data_root, tables, gold)
    result = {
        "status": "passed", "seeds": [20261020, 42], "questions_per_dataset": 1000,
        "official_release_hashes_verified": True, "ordered_ids_match_published_protocols": True,
        "historical_corpus_gold_and_membership_hashes_verified": True,
        "candidate_passages": {str(seed): values for seed, values in PASSAGES.items()},
        "data_root": str(data_root), "model_calls": 0, "network_requests": 0,
    }
    write_json(data_root / "preparation_validation.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
