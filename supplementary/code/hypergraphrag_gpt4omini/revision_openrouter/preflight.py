"""Offline full-input validation; no gold access and no remote API calls."""
from __future__ import annotations

from pathlib import Path
import tempfile

from hypergraphrag.operate import chunking_by_token_size
from hypergraphrag.utils import compute_mdhash_id
from .run import BASE, LOCK, CONFIG, document_text, prepare, read_json
from .transport import signature, write_json


def main():
    rows = []
    with tempfile.TemporaryDirectory() as temporary:
        for entry in read_json(LOCK)["datasets"]:
            root = Path(temporary) / entry["dataset"]
            _, corpus, manifest = prepare(entry, root, 1000)
            docs = [document_text(p).strip() for p in corpus]
            if len(set(docs)) != len(docs):
                raise RuntimeError("Duplicate document content")
            chunks = [
                compute_mdhash_id(chunk["content"], prefix="chunk-")
                for doc in docs
                for chunk in chunking_by_token_size(
                    doc, 100, 1200, tiktoken_model="gpt-4o-mini")
            ]
            if len(set(chunks)) != len(chunks):
                raise RuntimeError("Duplicate chunk content")
            rows.append({
                "dataset": entry["dataset"], "questions": manifest["question_count"],
                "passages": len(docs), "chunks": len(chunks),
                "corpus_sha256": manifest["corpus_sha256"],
                "queries_sha256": manifest["queries_sha256"],
            })
    record = {
        "status": "passed", "config_sha256": signature(CONFIG), "datasets": rows,
        "gold_loaded": False, "remote_api_calls": 0,
    }
    write_json(BASE / "revision_openrouter/preflight_validation.json", record)
    print(record)


if __name__ == "__main__":
    main()
