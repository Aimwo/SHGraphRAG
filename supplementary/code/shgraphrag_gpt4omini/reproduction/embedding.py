"""BGE-M3 through the existing local Ollama service; exact cosine indexes."""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import threading
import time

import numpy as np
import requests

from .client import append_jsonl, signature, write_json


MODEL = "bge-m3:latest"
ENDPOINT = "http://127.0.0.1:11434"


class Embeddings:
    def __init__(self, root: Path, *, run_id: str | None = None):
        root.mkdir(parents=True, exist_ok=True)
        self.root = root
        self.run_id = run_id
        self.session = requests.Session()
        self.session.trust_env = False
        tags = self.session.get(ENDPOINT + "/api/tags", timeout=15).json()
        self.model = next(m for m in tags["models"] if m["name"] == MODEL)
        self.digest = self.model["digest"]
        self.lock = threading.Lock()
        self.db = sqlite3.connect(root / "embedding_cache.sqlite3",
                                  check_same_thread=False, timeout=120)
        self.db.execute("CREATE TABLE IF NOT EXISTS embeddings "
                        "(key TEXT PRIMARY KEY, vector BLOB NOT NULL)")
        write_json(root / "embedding_model.json", self.model)

    def encode(self, texts: list[str], *, stage: str, cache: bool = True) -> np.ndarray:
        if not texts:
            return np.empty((0, 1024), dtype=np.float32)
        keys = [signature({"model_digest": self.digest, "text": t}) for t in texts]
        result = []
        with self.lock:
            # Serializing this service avoids competing embedding batches and
            # keeps SQLite writes and per-query inference deterministic.
            missing = {}
            known = {}
            for key, text in zip(keys, texts):
                if key in known or key in missing:
                    continue
                row = (self.db.execute("SELECT vector FROM embeddings WHERE key=?",
                                       (key,)).fetchone() if cache else None)
                if row:
                    known[key] = np.frombuffer(row[0], dtype=np.float32).copy()
                else:
                    missing[key] = text
            hit_keys = set(known)
            remaining = list(missing)
            for start in range(0, len(remaining), 32):
                batch_keys = remaining[start:start + 32]
                began = time.perf_counter()
                began_unix = time.time()
                payload = {"model": MODEL, "input": [missing[k] for k in batch_keys],
                           "truncate": False, "keep_alive": "5m"}
                response = None
                try:
                    response = self.session.post(
                        ENDPOINT + "/api/embed", json=payload, timeout=(15, 180))
                    response.raise_for_status()
                except requests.RequestException as error:
                    append_jsonl(self.root / "embedding_failures.jsonl", {
                        "stage": stage, "run_id": self.run_id,
                        "started_unix": began_unix,
                        "elapsed_s": time.perf_counter() - began,
                        "batch": len(batch_keys), "model_digest": self.digest,
                        "request_sha256": signature(payload),
                        "input_digest_keys": batch_keys,
                        "http_status": (response.status_code
                                        if response is not None else None),
                        "local_service_error_body": (response.text[:4000]
                                                     if response is not None else None),
                        "error_type": type(error).__name__,
                        "retry_policy": "no_automatic_retry",
                    })
                    raise
                data = response.json()
                vectors = np.asarray(data["embeddings"], dtype=np.float32)
                if (vectors.shape != (len(batch_keys), 1024)
                        or not np.isfinite(vectors).all()):
                    raise RuntimeError("BGE-M3 output dimension/nonfinite failure")
                norms = np.linalg.norm(vectors, axis=1)
                if np.any(norms < 0.1):
                    raise RuntimeError("BGE-M3 returned an invalid zero vector")
                vectors /= norms[:, None]
                for key, vector in zip(batch_keys, vectors):
                    known[key] = vector
                    if cache:
                        self.db.execute("INSERT OR REPLACE INTO embeddings VALUES (?,?)",
                                        (key, vector.tobytes()))
                self.db.commit()
                append_jsonl(self.root / "embedding_calls.jsonl", {
                    "stage": stage, "batch": len(batch_keys),
                    "run_id": self.run_id, "started_unix": began_unix,
                    "model_digest": self.digest,
                    "elapsed_s": time.perf_counter() - began,
                    "prompt_eval_count": data.get("prompt_eval_count"),
                    "total_duration_ns": data.get("total_duration"),
                    "load_duration_ns": data.get("load_duration"),
                    "keep_alive": "5m",
                })
            append_jsonl(self.root / "embedding_cache_usage.jsonl", {
                "stage": stage, "run_id": self.run_id, "observed_unix": time.time(),
                "model_digest": self.digest, "cache_enabled": cache,
                "input_rows": len(keys), "distinct_input_texts": len(set(keys)),
                "database_hit_distinct_texts": len(hit_keys),
                "database_hit_rows": sum(key in hit_keys for key in keys),
                "newly_encoded_distinct_texts": len(remaining),
                "duplicate_rows_within_call": len(keys) - len(set(keys)),
                "cache_key": "SHA256(canonical({model_digest,text}))",
            })
            result = [known[k] for k in keys]
        return np.stack(result)


class VectorIndex:
    def __init__(self, items: list[dict], vectors: np.ndarray):
        if len(items) != len(vectors):
            raise ValueError("Index item/vector count mismatch")
        self.items = items
        self.vectors = vectors

    @classmethod
    def build(cls, path: Path, items: list[dict], texts: list[str],
              embeddings: Embeddings, *, stage: str) -> "VectorIndex":
        fingerprint = signature({"items": items, "texts": texts,
                                 "digest": embeddings.digest})
        meta = path.with_suffix(".json")
        matrix = path.with_suffix(".npy")
        if meta.exists() and matrix.exists():
            data = json.loads(meta.read_text(encoding="utf-8"))
            if data.get("fingerprint") == fingerprint:
                return cls(data["items"], np.load(matrix, mmap_mode="r"))
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".partial.npy")
        checkpoint = path.with_suffix(".checkpoint.json")
        count = len(items)
        done = 0
        if partial.exists() and checkpoint.exists():
            state = json.loads(checkpoint.read_text(encoding="utf-8"))
            if state["fingerprint"] != fingerprint:
                raise RuntimeError("Partial index belongs to different inputs")
            done = state["completed"]
            vectors = np.lib.format.open_memmap(partial, mode="r+")
        else:
            vectors = np.lib.format.open_memmap(
                partial, mode="w+", dtype=np.float32, shape=(count, 1024))
        for start in range(done, count, 1024):
            began = time.perf_counter()
            stop = min(start + 1024, count)
            vectors[start:stop] = embeddings.encode(
                texts[start:stop], stage=stage, cache=True)
            vectors.flush()
            write_json(checkpoint, {"fingerprint": fingerprint, "completed": stop,
                                    "total": count})
            append_jsonl(path.parent / "index_timing.jsonl", {
                "stage": stage, "start": start, "stop": stop,
                "vectors": stop - start, "elapsed_s": time.perf_counter() - began,
                "completed_unix": time.time()})
        vectors.flush()
        del vectors
        partial.replace(matrix)
        write_json(meta, {"fingerprint": fingerprint, "items": items,
                         "model_digest": embeddings.digest,
                         "dimensions": 1024, "normalized": True})
        return cls(items, np.load(matrix, mmap_mode="r"))

    def search(self, vector: np.ndarray, k: int,
               allowed: set[str] | None = None) -> list[dict]:
        if not len(self.items):
            return []
        scores = self.vectors @ vector
        indexes = np.arange(len(self.items))
        if allowed is not None:
            indexes = np.array([i for i, item in enumerate(self.items)
                                if item["id"] in allowed], dtype=np.int64)
        if not len(indexes):
            return []
        k = min(k, len(indexes))
        # Stable tie-breaking makes retrieval repeatable.
        picked = indexes[np.argsort(-scores[indexes], kind="stable")[:k]]
        return [{**self.items[int(i)], "score": float(scores[i])} for i in picked]
