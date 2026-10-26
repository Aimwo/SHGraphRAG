"""Exact-request caches, one configured 100-slot gate, and complete per-attempt evidence."""
from __future__ import annotations

import asyncio
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sqlite3
import time
import uuid

import httpx
import numpy as np

from hypergraphrag.utils import LLMIncompleteError, LLMRefusalError, LLMTruncationError
from .common import (
    ITEM, append_jsonl, canonical, file_hash, read_json, signature, write_json,
)
from .config import validate
from .capacity_recovery import require_model_calls_allowed


class ProviderError(RuntimeError):
    pass


def retry_status(code):
    if isinstance(code, str) and code.isdigit():
        code = int(code)
    return isinstance(code, int) and (code in (408, 409, 425, 429) or 500 <= code <= 599)


def usage_counts(raw):
    """Missing counters remain unknown; cached/reasoning counts are subsets."""
    raw = raw if isinstance(raw, dict) else {}
    prompt_details = raw.get("prompt_tokens_details") or raw.get("input_tokens_details") or {}
    completion_details = raw.get("completion_tokens_details") or raw.get("output_tokens_details") or {}
    return {
        "input_tokens": raw.get("prompt_tokens", raw.get("input_tokens")),
        "output_tokens": raw.get("completion_tokens", raw.get("output_tokens")),
        "total_tokens": raw.get("total_tokens"),
        "cached_input_tokens": prompt_details.get("cached_tokens", raw.get("prompt_cache_hit_tokens")),
        "reasoning_output_tokens": completion_details.get("reasoning_tokens"),
    }


def provider_identity(config):
    return {
        "provider": config["provider"], "base_url": config["base_url"],
        "model": config["model"], "credential_id": config["credential_id"],
        "generation_parameters": config["generation_parameters"],
        "allowed_returned_models": config["allowed_returned_models"],
        "embedding": config["embedding"],
    }


class Transport:
    def __init__(self, config, cache, log_root, *, key, text_client=None, embedding_client=None):
        validate(config)
        if not key:
            raise ValueError("The explicitly selected task credential is required")
        self.config, self.cache, self.log_root = config, Path(cache), Path(log_root)
        self._key = key
        (self.cache / "responses").mkdir(parents=True, exist_ok=True)
        (self.cache / "attempts").mkdir(exist_ok=True)
        kwargs = dict(trust_env=False, follow_redirects=False,
                      timeout=httpx.Timeout(config["request_timeout_seconds"], connect=30),
                      limits=httpx.Limits(max_connections=config["max_in_flight"],
                                         max_keepalive_connections=config["max_in_flight"]))
        self.text_client = text_client or httpx.AsyncClient(proxy=config.get("proxy"), **kwargs)
        self.embedding_client = embedding_client or httpx.AsyncClient(**kwargs)
        self.semaphore = asyncio.Semaphore(config["max_in_flight"])
        self.embed_semaphore = asyncio.Semaphore(config["embedding"]["max_in_flight"])
        self.locks = {}
        self.identity = provider_identity(config)
        self.scope_hash = signature(self.identity)
        self.db = sqlite3.connect(self.cache / "embeddings.sqlite")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, vector BLOB NOT NULL)")
        self.calls = self.hits = self.embedding_calls = 0
        self.active_calls = self.peak_active_calls = self.active_requests = self.peak_requests = 0
        self.stages = Counter()
        self.item_counts = Counter()
        self.extracted_chunks = defaultdict(set)

    def redact(self, value):
        if isinstance(value, dict):
            return {k: ("[REDACTED]" if k.lower() in (
                "authorization", "api_key", "encrypted_key") else self.redact(v))
                for k, v in value.items()}
        if isinstance(value, list):
            return [self.redact(v) for v in value]
        if isinstance(value, str):
            safe = value.replace(self._key, "[REDACTED_KEY]") if self._key else value
            return safe
        return value

    async def close(self):
        self.db.close()
        await self.text_client.aclose()
        await self.embedding_client.aclose()
        self._key = ""

    async def _exchange(self, kind, body, stage, validator):
        require_model_calls_allowed()
        context = dict(ITEM.get())
        is_text = kind == "text"
        endpoint = (self.config["base_url"].rstrip("/") + "/chat/completions" if is_text
                    else self.config["embedding"]["base_url"] + "/api/embed")
        client = self.text_client if is_text else self.embedding_client
        headers = {"Authorization": "Bearer " + self._key} if is_text else {}
        request_hash = signature({"scope": self.scope_hash, "endpoint": endpoint, "body": body})
        for attempt in range(1, self.config["max_text_attempts"] + 1):
            queued = time.perf_counter()
            terminal = None
            retryable = False
            require_model_calls_allowed()
            async with self.semaphore:
                require_model_calls_allowed()
                queue_s = time.perf_counter() - queued
                started = time.perf_counter()
                attempt_id = uuid.uuid4().hex
                record = {
                    **context, "kind": kind, "stage": stage, "attempt": attempt,
                    "attempt_id": attempt_id, "request_sha256": request_hash,
                    "started_unix": time.time(), "queue_s": queue_s,
                    "endpoint": endpoint, "requested_model": body["model"],
                    "credential_id": self.config["credential_id"] if is_text else None,
                    "scope_sha256": self.scope_hash,
                    "raw_artifact": "cache/attempts/" + attempt_id + ".json",
                    "raw_usage": None,
                }
                self.active_requests += 1
                self.peak_requests = max(self.peak_requests, self.active_requests)
                if is_text:
                    self.active_calls += 1
                    self.peak_active_calls = max(self.peak_active_calls, self.active_calls)
                    self.calls += 1
                    self.item_counts[(context.get("dataset"), context.get("item_id"), "calls")] += 1
                else:
                    self.embedding_calls += 1
                    self.item_counts[(context.get("dataset"), context.get("item_id"), "embedding_requests")] += 1
                record["active_requests_on_send"] = self.active_requests
                record["active_text_calls_on_send"] = self.active_calls
                record["active_embedding_requests_on_send"] = self.active_requests - self.active_calls
                artifact = {"request": body, "endpoint": endpoint, "response_text": None}
                try:
                    # Persist the original task input before sending. A hard interruption
                    # leaves explicit incomplete evidence instead of hiding a request.
                    write_json(self.cache / "attempts" / (attempt_id + ".json"),
                               self.redact({**artifact, "record": {**record, "status": "in_flight"}}))
                    append_jsonl(self.log_root / "request_started.jsonl", self.redact(record))
                    response = await client.post(endpoint, json=body, headers=headers)
                    record["http_status"] = response.status_code
                    artifact["response_text"] = response.text
                    try:
                        data = response.json()
                    except ValueError:
                        data = None
                    artifact["response_json"] = data
                    if isinstance(data, dict):
                        record["raw_usage"] = data.get("usage")
                        record["returned_model"] = data.get("model")
                        record["response_id"] = data.get("id")
                        if not is_text:
                            record["embedding_native_usage"] = {
                                k: data.get(k) for k in ("prompt_eval_count", "total_duration", "load_duration")}
                    if response.status_code != 200:
                        retryable = retry_status(response.status_code)
                        if not is_text and response.status_code == 400:
                            error = str((data or {}).get("error", ""))
                            retryable = ("/tokenize" in error and "127.0.0.1:" in error
                                         and ("refused" in error.lower()
                                              or "lacked sufficient buffer space" in error.lower()))
                        raise ProviderError("http_" + str(response.status_code))
                    if not isinstance(data, dict):
                        retryable = True
                        raise ProviderError("invalid_response_JSON")
                    if data.get("error"):
                        error = data["error"]
                        retryable = retry_status(error.get("code")) if isinstance(error, dict) else False
                        raise ProviderError("provider_error_response")
                    result = validator(data, request_hash)
                    record["status"] = "completed"
                    self.stages[stage + "_completed"] += 1
                    artifact["outcome"] = "completed"
                except (LLMIncompleteError, LLMRefusalError, LLMTruncationError) as exc:
                    record.update(status=exc.outcome, error=str(exc))
                    artifact["outcome"] = exc.outcome
                    terminal = exc
                except (httpx.TransportError, ValueError) as exc:
                    retryable = True
                    record.update(status="failed", error_type=type(exc).__name__)
                    terminal = ProviderError(type(exc).__name__)
                except ProviderError as exc:
                    record.update(status="failed", error=str(exc))
                    terminal = exc
                except BaseException as exc:
                    record.update(status="cancelled" if isinstance(exc, asyncio.CancelledError)
                                  else "failed", error_type=type(exc).__name__)
                    terminal = exc
                finally:
                    self.active_requests -= 1
                    if is_text:
                        self.active_calls -= 1
                    record.update(finished_unix=time.time(), elapsed_s=time.perf_counter() - started,
                                  retryable=retryable, usage=usage_counts(record["raw_usage"]))
                    record["raw_payload_sha256"] = signature(self.redact(artifact))
                    write_json(self.cache / "attempts" / (attempt_id + ".json"),
                               self.redact({**artifact, "record": record}))
                    append_jsonl(self.log_root / ("api_calls.jsonl" if is_text
                                                 else "embedding_calls.jsonl"), self.redact(record))
                if terminal is None:
                    return result, record
            if not retryable or attempt == self.config["max_text_attempts"]:
                raise terminal
            backoff = min(2 ** (attempt - 1), 4)
            append_jsonl(self.log_root / "retry_events.jsonl", {
                **context, "kind": kind, "request_sha256": request_hash,
                "previous_attempt_id": attempt_id, "backoff_s": backoff})
            await asyncio.sleep(backoff)
        raise AssertionError("unreachable")

    async def complete(self, prompt, system_prompt=None, history_messages=None,
                       max_tokens=None, **kwargs):
        if kwargs.get("stream"):
            raise ValueError("The audit saves complete non-streaming responses")
        if "knowledge segments" in prompt or "-Goal-" in prompt:
            stage = "query_extract" if ITEM.get().get("item_id") else "extract"
        elif "MANY knowdge" in prompt:
            stage = "glean"
        elif "whether knowdge" in prompt:
            stage = "coverage"
        elif "comprehensive summary" in prompt:
            stage = "summary"
        else:
            stage = "answer" if system_prompt else ITEM.get().get("stage", "llm")
        messages = ([{"role": "system", "content": system_prompt}] if system_prompt else [])
        messages += list(history_messages or [])
        messages.append({"role": "user", "content": prompt})
        body = {"model": self.config["model"], "messages": messages, "stream": False,
                **self.config["generation_parameters"]}
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        for key in ("temperature", "top_p", "response_format"):
            if key in kwargs:
                body[key] = kwargs[key]
        cache_key = signature({"scope": self.scope_hash, "kind": "text", "body": body})
        path = self.cache / "responses" / (cache_key + ".json")
        async with self.locks.setdefault(cache_key, asyncio.Lock()):
            if path.exists():
                saved = read_json(path)
                if saved["cache_key"] != cache_key or saved["scope_sha256"] != self.scope_hash:
                    raise ProviderError("cache_identity_mismatch")
                content = saved["text"] if saved["status"] == "completed" else saved["raw_text"]
                if saved.get("content_sha256") != signature(content):
                    raise ProviderError("cache_content_hash_mismatch")
                self.hits += 1
                self.item_counts[(ITEM.get().get("dataset"), ITEM.get().get("item_id"), "hits")] += 1
                append_jsonl(self.log_root / "cache_hits.jsonl", {
                    **ITEM.get(), "kind": "text", "stage": stage, "cache_key": cache_key})
                if saved["status"] != "completed":
                    self._raise_unavailable(saved["status"], saved["raw_text"],
                                            saved["request_sha256"], saved.get("response_id"))
                return saved["text"]

            def parse(data, request_hash):
                if data.get("model") not in self.config["allowed_returned_models"]:
                    raise ProviderError("returned_model_not_authorized")
                choices = data.get("choices")
                if not isinstance(choices, list) or len(choices) != 1:
                    raise ProviderError("expected_one_completion_choice")
                choice = choices[0]
                message = choice.get("message") or {}
                raw = message.get("content") or ""
                if not isinstance(raw, str):
                    raise ProviderError("nontext_completion")
                finish = choice.get("finish_reason")
                if message.get("refusal") or finish == "content_filter":
                    self._raise_unavailable("provider_filtered", raw, request_hash, data.get("id"))
                if finish == "length":
                    self._raise_unavailable("output_truncated", raw, request_hash, data.get("id"))
                if finish != "stop":
                    self._raise_unavailable("output_incomplete", raw, request_hash, data.get("id"))
                if not raw.strip():
                    raise ProviderError("empty_completed_text")
                return raw

            try:
                text, record = await self._exchange("text", body, stage, parse)
            except (LLMIncompleteError, LLMRefusalError, LLMTruncationError) as exc:
                write_json(path, {"cache_key": cache_key, "scope_sha256": self.scope_hash,
                                 "status": exc.outcome, "raw_text": self.redact(exc.raw_text),
                                 "content_sha256": signature(self.redact(exc.raw_text)),
                                 "request_sha256": exc.request_sha256, "response_id": exc.response_id})
                raise
            write_json(path, {"cache_key": cache_key, "scope_sha256": self.scope_hash,
                             "status": "completed", "text": self.redact(text),
                             "content_sha256": signature(self.redact(text)),
                             "request_sha256": record["request_sha256"],
                             "response_id": record.get("response_id"),
                             "raw_artifact": record["raw_artifact"]})
            return self.redact(text)

    @staticmethod
    def _raise_unavailable(outcome, raw, request_hash, response_id):
        cls = {"provider_filtered": LLMRefusalError, "output_truncated": LLMTruncationError,
               "output_incomplete": LLMIncompleteError}[outcome]
        raise cls(outcome, request_hash, raw, response_id)

    async def embed(self, texts):
        if not texts:
            return np.empty((0, 1024), dtype=np.float32)
        async with self.embed_semaphore:
            keys = [signature({"scope": self.scope_hash, "kind": "embedding", "text": t})
                    for t in texts]
            result, missing = [], {}
            for i, key in enumerate(keys):
                row = self.db.execute("SELECT vector FROM vectors WHERE key=?", (key,)).fetchone()
                result.append(np.frombuffer(row[0], dtype="<f4").copy() if row else None)
                if row is None:
                    missing.setdefault(key, (texts[i], []))[1].append(i)
            items = list(missing.items())
            append_jsonl(self.log_root / "cache_hits.jsonl", {
                **ITEM.get(), "kind": "embedding", "inputs": len(texts),
                "cached_input_positions": sum(v is not None for v in result),
                "unique_uncached_texts": len(items)})
            for offset in range(0, len(items), 32):
                batch = items[offset:offset + 32]
                # Preserve HTTP batching; use this recovery's explicit local capacity.
                body = {"model": "bge-m3:latest", "input": [v[0] for _, v in batch],
                        "options": dict(self.config["embedding"]["options"])}

                def parse(data, _request_hash):
                    if data.get("model") not in (None, "bge-m3", "bge-m3:latest"):
                        raise ProviderError("unexpected_local_embedding_model")
                    vectors = np.asarray(data.get("embeddings"), dtype=np.float32)
                    if vectors.shape != (len(batch), 1024) or not np.isfinite(vectors).all():
                        raise ProviderError("invalid_BGE_vector_shape_or_values")
                    if np.any(np.linalg.norm(vectors, axis=1) < 1e-8):
                        raise ProviderError("zero_BGE_vector")
                    return vectors

                vectors, _record = await self._exchange(
                    "embedding", body, ITEM.get().get("stage", "embedding"), parse)
                for (key, (_, indices)), vector in zip(batch, vectors):
                    self.db.execute("INSERT OR IGNORE INTO vectors VALUES (?,?)",
                                    (key, vector.astype("<f4").tobytes()))
                    for i in indices:
                        result[i] = vector.copy()
                self.db.commit()
            return np.stack(result)
