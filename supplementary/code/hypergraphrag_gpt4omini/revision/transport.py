"""Resumable transports. Credentials are never serialized or logged."""
from __future__ import annotations

import asyncio
import contextvars
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import time

import httpx
import numpy as np

MODEL = "gpt-6-luna"
BASE_URL = "https://www.fucheers.top/v1"
KEY_FILE = None  # Legacy provider intentionally unavailable in this release.
EMBED_MODEL = "bge-m3:latest"
EMBED_DIGEST = "7907646426070047a77226ac3e684fbbe8410524f7b4a74d02837e43f2146bab"
ITEM = contextvars.ContextVar("hypergraph_item", default={})


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def signature(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    descriptor, name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
        for attempt in range(8):
            try:
                temporary.replace(path)
                return
            except PermissionError as exc:
                # Ordinary Windows readers briefly deny rename/delete sharing.
                # Retry the identical serialized bytes; permanent errors escape.
                if (os.name != "nt" or getattr(exc, "winerror", None) not in (5, 32, 33)
                        or attempt == 7):
                    raise
                time.sleep(min(0.02 * 2 ** attempt, 0.5))
    finally:
        temporary.unlink(missing_ok=True)


def append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(canonical(value) + "\n")


class ProviderError(RuntimeError):
    pass


class Transport:
    def __init__(self, cache: Path, log_root: Path, workers=1):
        self.cache, self.log_root = cache, log_root
        (cache / "responses").mkdir(parents=True, exist_ok=True)
        self.client = httpx.AsyncClient(
            trust_env=False, follow_redirects=False,
            timeout=httpx.Timeout(240, connect=20),
            limits=httpx.Limits(max_connections=max(4, workers + 1)))
        self.semaphore = asyncio.Semaphore(workers)
        self.embed_semaphore = asyncio.Semaphore(1)
        self.locks = {}
        raise RuntimeError("Legacy provider is disabled; use revision_openrouter.Transport")
        self._key = ""
        if not self._key:
            raise ProviderError("empty_shared_key")
        self.db = sqlite3.connect(cache / "embeddings.sqlite")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS vectors "
                        "(key TEXT PRIMARY KEY, vector BLOB NOT NULL)")
        self.calls = self.hits = self.embedding_calls = 0
        self.stages = Counter()
        self.item_counts = Counter()
        self.extracted_chunks = defaultdict(set)

    async def close(self):
        self.db.close()
        await self.client.aclose()

    async def check_embedding(self):
        response = await self.client.get("http://127.0.0.1:11434/api/tags")
        response.raise_for_status()
        models = {m["name"]: m for m in response.json()["models"]}
        found = models.get(EMBED_MODEL)
        if not found or found.get("digest") != EMBED_DIGEST:
            raise ProviderError("BGE_model_digest_mismatch")
        vector = await self.embed(["HyperGraphRAG local embedding contract."])
        running = await self.client.get("http://127.0.0.1:11434/api/ps")
        running.raise_for_status()
        loaded = [m for m in running.json().get("models", [])
                  if m.get("name") == EMBED_MODEL]
        return {"model": EMBED_MODEL, "digest": found["digest"],
                "dimensions": vector.shape[1], "norm": float(np.linalg.norm(vector[0])),
                "manifest_details": found.get("details"),
                "loaded_model_at_start": loaded}

    async def complete(self, prompt, system_prompt=None, history_messages=None,
                       max_tokens=None, **kwargs):
        if kwargs.get("stream"):
            raise ValueError("This evaluation records complete non-streaming responses")
        context = dict(ITEM.get())
        # Classify calls without altering the original method prompts.
        if "knowledge segments" in prompt or "-Goal-" in prompt:
            stage = "extract"
        elif "MANY knowdge" in prompt:
            stage = "glean"
        elif "whether knowdge" in prompt:
            stage = "coverage"
        elif "comprehensive summary" in prompt:
            stage = "summarize"
        elif system_prompt:
            stage = "answer"
        else:
            stage = context.get("stage", "llm")
        if stage == "extract" and context.get("item_id"):
            stage = "query_extract"
        if stage in ("extract", "glean", "coverage"):
            initial = (history_messages[0]["content"] if history_messages else prompt)
            marker = "-Real Data-\n######################\nText: "
            tail = "\n######################\nOutput:"
            if marker in initial and tail in initial:
                content = initial.split(marker, 1)[1].rsplit(tail, 1)[0]
                context["chunk_id"] = "chunk-" + hashlib.md5(content.encode()).hexdigest()
        cap = max_tokens or {"extract": 8192, "glean": 8192, "coverage": 64,
                             "query_extract": 8192,
                             "answer": 512, "summarize": 500}.get(stage, 4096)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history_messages or [])
        messages.append({"role": "user", "content": prompt})
        body = {"model": MODEL, "reasoning": {"effort": "none"},
                "input": messages, "max_output_tokens": cap,
                "stream": False, "store": False}
        key = signature(body)
        path = self.cache / "responses" / (key + ".json")
        async with self.locks.setdefault(key, asyncio.Lock()):
            if path.exists():
                saved = json.loads(path.read_text(encoding="utf-8"))
                if saved["request_sha256"] != key or saved["model"] != MODEL:
                    raise ProviderError("response_cache_identity_mismatch")
                self.hits += 1
                self.item_counts[(context.get("dataset"), context.get("item_id"), "hits")] += 1
                self.stages[stage + "_cache_hits"] += 1
                if stage == "extract" and context.get("chunk_id"):
                    self.extracted_chunks[context.get("dataset")].add(context["chunk_id"])
                append_jsonl(self.log_root / "cache_hits.jsonl",
                             {**context, "stage": stage, "request_sha256": key,
                              "time": time.time()})
                return saved["text"]
            async with self.semaphore:
                for attempt in range(1, 5):
                    start = time.perf_counter()
                    record = {**context, "stage": stage, "attempt": attempt,
                              "request_sha256": key, "started_unix": time.time(),
                              "model": MODEL, "reasoning_effort": "none"}
                    retryable = False
                    try:
                        response = await self.client.post(
                            BASE_URL + "/responses", json=body,
                            headers={"Authorization": "Bearer " + self._key})
                        record["http_status"] = response.status_code
                        if response.status_code != 200:
                            retryable = (response.status_code in (408, 409, 425, 429)
                                         or response.status_code >= 500)
                            raise ProviderError("http_" + str(response.status_code))
                        data = response.json()
                        record["usage"] = data.get("usage", {})
                        if data.get("model") != MODEL:
                            raise ProviderError("returned_model_mismatch")
                        if data.get("status") != "completed":
                            raise ProviderError("response_" + str(data.get("status")))
                        text = "".join(part.get("text", "")
                                       for item in data.get("output", [])
                                       for part in item.get("content", [])
                                       if part.get("type") == "output_text")
                        if not text.strip():
                            retryable = True
                            raise ProviderError("empty_output_text")
                        saved = {"request_sha256": key, "request": body,
                                 "model": data["model"], "status": data["status"],
                                 "text": text, "usage": data.get("usage", {}),
                                 "response_id": data.get("id"),
                                 "elapsed_s": time.perf_counter() - start}
                        write_json(path, saved)
                        record.update(status="completed",
                                      elapsed_s=saved["elapsed_s"],
                                      response_id=saved["response_id"])
                        append_jsonl(self.log_root / "api_calls.jsonl", record)
                        self.calls += 1
                        self.item_counts[(context.get("dataset"), context.get("item_id"), "calls")] += 1
                        self.stages[stage + "_completed"] += 1
                        if stage == "extract" and context.get("chunk_id"):
                            self.extracted_chunks[context.get("dataset")].add(context["chunk_id"])
                        return text
                    except (httpx.TransportError, json.JSONDecodeError) as exc:
                        retryable = True
                        error = type(exc).__name__
                    except ProviderError as exc:
                        error = str(exc)
                    record.update(status="failed", error=error,
                                  elapsed_s=time.perf_counter() - start)
                    append_jsonl(self.log_root / "api_calls.jsonl", record)
                    if not retryable or attempt == 4:
                        raise ProviderError(error)
                    await asyncio.sleep(min(30, 2 ** attempt))
        raise AssertionError("unreachable")

    async def embed(self, texts):
        keys = [signature([EMBED_MODEL, EMBED_DIGEST, text]) for text in texts]
        result = []
        missing = {}
        for index, key in enumerate(keys):
            row = self.db.execute("SELECT vector FROM vectors WHERE key=?", (key,)).fetchone()
            result.append(np.frombuffer(row[0], dtype="<f4").copy() if row else None)
            if not row:
                missing.setdefault(key, (texts[index], []))[1].append(index)
        items = list(missing.items())
        for offset in range(0, len(items), 16):
            batch = items[offset:offset + 16]
            start = time.perf_counter()
            async with self.embed_semaphore:
                response = await self.client.post(
                    "http://127.0.0.1:11434/api/embed",
                    json={"model": EMBED_MODEL, "input": [v[0] for _, v in batch],
                          "truncate": False, "keep_alive": "30m"})
                response.raise_for_status()
                data = response.json()
                vectors = np.asarray(data["embeddings"], dtype=np.float32)
                if vectors.shape != (len(batch), 1024) or not np.isfinite(vectors).all():
                    raise ProviderError("invalid_BGE_vectors")
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                if np.any(norms < 1e-8):
                    raise ProviderError("zero_BGE_vector")
                vectors /= norms
                for (key, (_, indices)), vector in zip(batch, vectors):
                    self.db.execute("INSERT OR IGNORE INTO vectors VALUES (?,?)",
                                    (key, vector.astype("<f4").tobytes()))
                    for index in indices:
                        result[index] = vector.copy()
                self.db.commit()
            self.embedding_calls += 1
            append_jsonl(self.log_root / "embedding_calls.jsonl", {
                **ITEM.get(), "model": EMBED_MODEL, "digest": EMBED_DIGEST,
                "texts": len(batch), "elapsed_s": time.perf_counter() - start,
                "prompt_eval_count": data.get("prompt_eval_count"),
                "total_duration_ns": data.get("total_duration")})
        return np.stack(result) if result else np.empty((0, 1024), dtype=np.float32)
