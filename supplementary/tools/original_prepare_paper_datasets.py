"""Prepare traceable development gold, frozen samples, and candidate corpora."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
READY = ROOT / "dataset" / "paper_ready"
SEED = 20261020
DATASETS = {
    "hotpotqa": ("dataset/hotpot_qa/distractor/validation-00000-of-00001.parquet", 7405),
    "2wiki": ("dataset/raw_official/2wiki/release_20210407/dev.json", 12576),
    "musique": ("dataset/raw_official/musique/release_v1/data/musique_ans_v1.0_dev.jsonl", 2417),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False)
        stream.write("\n")
    temporary.replace(path)


def write_jsonl(path: Path, rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    temporary.replace(path)
    return count


def hotpot_native(row: dict) -> dict:
    return {
        "_id": row["id"], "question": row["question"], "answer": row["answer"],
        "type": row["type"], "level": row["level"],
        "supporting_facts": [
            [title, index] for title, index in
            zip(row["supporting_facts"]["title"], row["supporting_facts"]["sent_id"])
        ],
        "context": [
            [title, sentences] for title, sentences in
            zip(row["context"]["title"], row["context"]["sentences"])
        ],
    }


def two_wiki_parquet_row(row: dict) -> dict:
    return {
        "id": row["_id"], "question": row["question"], "answer": row["answer"],
        "type": row["type"], "evidences": row["evidences"],
        "supporting_facts": {
            "title": [fact[0] for fact in row["supporting_facts"]],
            "sent_id": [fact[1] for fact in row["supporting_facts"]],
        },
        "context": {
            "title": [passage[0] for passage in row["context"]],
            "sentences": [passage[1] for passage in row["context"]],
        },
    }


def musique_parquet_row(row: dict) -> dict:
    return {
        "id": row["id"], "question": row["question"], "answer": row["answer"],
        "answerable": row["answerable"], "type": "",
        "context": {
            "title": [paragraph["title"] for paragraph in row["paragraphs"]],
            # Preserve each complete original paragraph without resegmentation.
            "sentences": [[paragraph["paragraph_text"]] for paragraph in row["paragraphs"]],
        },
    }


def recover_historical_ids(valid_ids: set[str], dataset: str) -> list[dict]:
    groups = {}
    paths = list((ROOT / "qa/outputs").glob("predictions*.json"))
    paths += list((ROOT / "qa/backup").rglob("predictions.json"))
    for path in sorted(paths):
        prediction = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(prediction, dict) or not isinstance(prediction.get("answer"), dict):
            continue
        ids = set(prediction["answer"])
        if len(ids) < 500 or not ids.issubset(valid_ids):
            continue
        signature = hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()
        if signature not in groups:
            groups[signature] = {
                "dataset": dataset, "count": len(ids), "ids": sorted(ids),
                "id_set_sha256": signature, "sources": [],
                "status": "recovered_from_predictions_not_verified_as_paper_sample",
            }
        groups[signature]["sources"].append({
            "path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path),
        })
    return list(groups.values())


def candidate_corpus(rows: list[dict], destination: Path) -> dict:
    passages = {}
    memberships = []
    for row in rows:
        ids = []
        context = row["context"]
        for title, sentences in zip(context["title"], context["sentences"]):
            text = "\n".join(sentences)
            digest = hashlib.sha256((title + "\0" + text).encode("utf-8")).hexdigest()
            passage_id = "passage_" + digest
            passages.setdefault(passage_id, {
                "passage_id": passage_id, "title": title,
                "sentences": sentences, "text": text,
            })
            ids.append(passage_id)
        memberships.append({"question_id": row["id"], "passage_ids": ids})
    corpus_path = destination / "candidate_corpus.jsonl"
    membership_path = destination / "question_passages.jsonl"
    write_jsonl(corpus_path, (passages[key] for key in sorted(passages)))
    write_jsonl(membership_path, memberships)
    return {
        "scope": "candidate passages associated with this frozen 1000-question sample",
        "deduplication": "exact title and newline-joined sentence text; SHA-256 passage IDs",
        "unique_passages": len(passages),
        "candidate_occurrences": sum(len(row["passage_ids"]) for row in memberships),
        "corpus": corpus_path.relative_to(ROOT).as_posix(),
        "corpus_sha256": sha256(corpus_path),
        "question_passages": membership_path.relative_to(ROOT).as_posix(),
        "question_passages_sha256": sha256(membership_path),
    }


def prepare() -> dict:
    manifest = {
        "prepared_on": "2026-10-20", "sample_seed": SEED,
        "sampling": "Python random.Random(seed).sample(sorted unique IDs, 1000)",
        "sample_status": "new_revision_samples_not_claimed_as_original_paper_samples",
        "datasets": {},
    }
    if (READY / "manifest.json").exists():
        previous = json.loads((READY / "manifest.json").read_text(encoding="utf-8"))
        if "hotpot_input_repair" in previous:
            manifest["hotpot_input_repair"] = previous["hotpot_input_repair"]
    raw_musique = ROOT / DATASETS["musique"][0]
    musique_native = {}
    with raw_musique.open(encoding="utf-8-sig") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if row["id"] in musique_native:
                    raise ValueError("Duplicate MuSiQue source ID")
                musique_native[row["id"]] = row
    for dataset, (source_name, expected) in DATASETS.items():
        source = ROOT / source_name
        if dataset == "2wiki":
            original = json.loads(source.read_text(encoding="utf-8"))
            table = pa.Table.from_pylist([two_wiki_parquet_row(row) for row in original])
        elif dataset == "musique":
            table = pa.Table.from_pylist([musique_parquet_row(row)
                                         for row in musique_native.values()])
        else:
            table = pq.read_table(source)
        rows = table.to_pylist()
        by_id = {row["id"]: row for row in rows}
        assert len(rows) == len(by_id) == expected, f"Unexpected count/duplicate ID: {dataset}"
        assert all(row["question"] and row["answer"] and row["context"]["title"] for row in rows)
        assert all(len(row["context"]["title"]) == len(row["context"]["sentences"]) for row in rows)
        destination = READY / dataset
        destination.mkdir(parents=True, exist_ok=True)
        if dataset == "hotpotqa":
            native = {key: hotpot_native(row) for key, row in by_id.items()}
            full_gold = destination / "dev.json"
            write_json(full_gold, [native[row["id"]] for row in rows])
            repair_target = ROOT / "qa/inputs/hotpot_dev_distractor_v1.json"
            if repair_target.exists() and sha256(repair_target) != sha256(full_gold):
                previous_hash = sha256(repair_target)
                backup = ROOT / "output/paper_revision/2026-10-20/dataset_backups" / (
                    "hotpot_dev_distractor_v1.before_repair." + previous_hash[:16] + ".json")
                backup.parent.mkdir(parents=True, exist_ok=True)
                if not backup.exists():
                    shutil.copy2(repair_target, backup)
                temporary = repair_target.with_name(repair_target.name + ".repair.tmp")
                shutil.copyfile(full_gold, temporary)
                temporary.replace(repair_target)
                manifest["hotpot_input_repair"] = {
                    "path": repair_target.relative_to(ROOT).as_posix(),
                    "original_backup": backup.relative_to(ROOT).as_posix(),
                    "original_sha256": previous_hash, "new_sha256": sha256(repair_target),
                    "rows": len(rows),
                }
        elif dataset == "2wiki":
            native = {row["_id"]: row for row in original}
            assert set(native) == set(by_id), "2Wiki JSON and Parquet ID sets differ"
            full_gold = destination / "dev.json"
            write_json(full_gold, original)
        else:
            native = musique_native
            assert set(native) == set(by_id), "MuSiQue JSONL and Parquet ID sets differ"
            full_gold = destination / "dev.jsonl"
            shutil.copyfile(raw_musique, full_gold)
        sample_ids = random.Random(SEED).sample(sorted(by_id), 1000)
        sample_folder = destination / f"sample_1000_seed{SEED}"
        sample_folder.mkdir(parents=True, exist_ok=True)
        write_json(sample_folder / "ids.json", sample_ids)
        if dataset == "musique":
            sample_gold = sample_folder / "gold.jsonl"
            write_jsonl(sample_gold, (native[key] for key in sample_ids))
        else:
            sample_gold = sample_folder / "gold.json"
            write_json(sample_gold, [native[key] for key in sample_ids])
        indices = {row["id"]: i for i, row in enumerate(rows)}
        sample_parquet = sample_folder / "questions.parquet"
        pq.write_table(table.take([indices[key] for key in sample_ids]), sample_parquet)
        historical = recover_historical_ids(set(by_id), dataset)
        write_json(destination / "recovered_historical_id_sets.json", historical)
        result = {
            "source": source_name, "source_sha256": sha256(source), "dev_rows": len(rows),
            "dev_gold": full_gold.relative_to(ROOT).as_posix(),
            "dev_gold_sha256": sha256(full_gold),
            "sample_ids": (sample_folder / "ids.json").relative_to(ROOT).as_posix(),
            "sample_gold": sample_gold.relative_to(ROOT).as_posix(),
            "sample_parquet": sample_parquet.relative_to(ROOT).as_posix(),
            "sample_gold_sha256": sha256(sample_gold),
            "sample_parquet_sha256": sha256(sample_parquet),
            "sample_ids_sha256": sha256(sample_folder / "ids.json"),
            "sample_size": len(sample_ids),
            "sample_id_set_sha256": hashlib.sha256("\n".join(sorted(sample_ids)).encode()).hexdigest(),
            "historical_sets": [{"count": group["count"], "sha256": group["id_set_sha256"]}
                                for group in historical],
            "candidate_corpus": candidate_corpus([by_id[key] for key in sample_ids], sample_folder),
        }
        manifest["datasets"][dataset] = result
        if dataset == "2wiki":
            result["answer_aliases"] = "dataset/raw_official/2wiki/release_20210407/id_aliases.json"
            result["answer_aliases_sha256"] = sha256(ROOT / result["answer_aliases"])
            result["release"] = "official 2021-04-07 (IDs release)"
            result["legacy_context_warning"] = (
                "The supplied qa/inputs/2wiki_dev.json has different candidate content "
                "for 1908 questions. New revision samples use the official release.")
        print(json.dumps({"dataset": dataset, "dev_rows": len(rows), "sample_rows": len(sample_ids),
                          "candidate_passages": result["candidate_corpus"]["unique_passages"]}), flush=True)
    write_json(READY / "manifest.json", manifest)
    return manifest


def validate() -> None:
    manifest = json.loads((READY / "manifest.json").read_text(encoding="utf-8"))
    results = {}
    for dataset, info in manifest["datasets"].items():
        ids_path = ROOT / info["sample_ids"]
        ids = json.loads(ids_path.read_text(encoding="utf-8"))
        assert len(ids) == len(set(ids)) == 1000
        assert sha256(ids_path) == info["sample_ids_sha256"]
        assert hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest() == info["sample_id_set_sha256"]
        table = pq.read_table(ROOT / info["sample_parquet"])
        assert table.column("id").to_pylist() == ids
        assert sha256(ROOT / info["sample_parquet"]) == info["sample_parquet_sha256"]
        native_path = ROOT / info["sample_gold"]
        if native_path.suffix == ".jsonl":
            native = [json.loads(line) for line in native_path.read_text(encoding="utf-8").splitlines()]
            native_ids = [row["id"] for row in native]
        else:
            native = json.loads(native_path.read_text(encoding="utf-8"))
            native_ids = [row["_id"] for row in native]
        assert native_ids == ids, f"Gold/sample order mismatch: {dataset}"
        assert sha256(native_path) == info["sample_gold_sha256"]
        assert sha256(ROOT / info["dev_gold"]) == info["dev_gold_sha256"]
        source = ROOT / info["source"]
        if dataset == "hotpotqa":
            source_rows = pq.read_table(source).to_pylist()
            source_native = {row["id"]: hotpot_native(row) for row in source_rows}
        elif dataset == "2wiki":
            original = json.loads(source.read_text(encoding="utf-8"))
            source_rows = [two_wiki_parquet_row(row) for row in original]
            source_native = {row["_id"]: row for row in original}
            assert sha256(ROOT / info["answer_aliases"]) == info["answer_aliases_sha256"]
        else:
            original = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()
                        if line.strip()]
            source_rows = [musique_parquet_row(row) for row in original]
            source_native = {row["id"]: row for row in original}
        source_by_id = {row["id"]: row for row in source_rows}
        assert len(source_by_id) == len(source_rows) == info["dev_rows"]
        dev_path = ROOT / info["dev_gold"]
        if dev_path.suffix == ".jsonl":
            dev_gold = [json.loads(line) for line in dev_path.read_text(encoding="utf-8").splitlines()
                        if line.strip()]
        else:
            dev_gold = json.loads(dev_path.read_text(encoding="utf-8"))
        assert dev_gold == [source_native[row["id"]] for row in source_rows]
        assert ids == random.Random(manifest["sample_seed"]).sample(sorted(source_by_id), 1000)
        assert table.to_pylist() == [source_by_id[key] for key in ids]
        assert native == [source_native[key] for key in ids]
        corpus_info = info["candidate_corpus"]
        corpus_file = ROOT / corpus_info["corpus"]
        assert sha256(corpus_file) == corpus_info["corpus_sha256"]
        passages = [json.loads(line) for line in corpus_file.read_text(encoding="utf-8").splitlines()]
        passage_by_id = {row["passage_id"]: row for row in passages}
        passage_ids = set(passage_by_id)
        assert len(passages) == len(passage_ids) == corpus_info["unique_passages"]
        for passage in passages:
            assert passage["text"] == "\n".join(passage["sentences"])
            digest = hashlib.sha256((passage["title"] + "\0" + passage["text"]).encode("utf-8")).hexdigest()
            assert passage["passage_id"] == "passage_" + digest
        with (ROOT / corpus_info["question_passages"]).open(encoding="utf-8") as stream:
            membership = [json.loads(line) for line in stream]
        assert [row["question_id"] for row in membership] == ids
        assert all(set(row["passage_ids"]).issubset(passage_ids) for row in membership)
        for question, mapping in zip(table.to_pylist(), membership):
            contexts = question["context"]
            expected = list(zip(contexts["title"], contexts["sentences"]))
            actual = [(passage_by_id[key]["title"], passage_by_id[key]["sentences"])
                      for key in mapping["passage_ids"]]
            assert actual == expected
        assert sha256(ROOT / corpus_info["question_passages"]) == corpus_info["question_passages_sha256"]
        assert sha256(source) == info["source_sha256"]
        results[dataset] = {
            "status": "passed", "sample_questions": len(ids),
            "unique_candidate_passages": len(passages),
            "candidate_occurrences": sum(len(row["passage_ids"]) for row in membership),
            "gold_and_parquet_match_canonical_source": True,
            "seed_and_sample_order_verified": True,
            "all_candidate_text_and_memberships_verified": True,
            "all_manifest_artifact_hashes_verified": True,
        }
        print(json.dumps({"dataset": dataset, "validation": "passed"}), flush=True)
    repaired = json.loads((ROOT / "qa/inputs/hotpot_dev_distractor_v1.json").read_text(encoding="utf-8"))
    assert len(repaired) == 7405 and len({row["_id"] for row in repaired}) == 7405
    repair = manifest.get("hotpot_input_repair")
    if repair:
        assert sha256(ROOT / repair["original_backup"]) == repair["original_sha256"]
        assert sha256(ROOT / repair["path"]) == repair["new_sha256"]
    write_json(READY / "validation.json", {
        "status": "passed", "validated_on": "2026-10-20", "datasets": results,
        "hotpot_original_backup_and_repaired_file_hashes_verified": bool(repair),
    })
    print("PASS: sample IDs, gold formats, Parquet order, source hashes and corpus references.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if not args.validate_only:
        prepare()
    validate()
