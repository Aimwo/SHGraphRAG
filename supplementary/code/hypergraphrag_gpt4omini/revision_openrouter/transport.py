"""OpenRouter text transport; loopback embeddings bypass all proxy settings."""
from __future__ import annotations

import asyncio
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import sqlite3
import time

import httpx
from .proxy_client import ProxyTextClient
from .embedding_client import (
    EMBED_OPTIONS, ENDPOINT, OVERFLOW_POLICY, LocalEmbeddingClient,
)
from hypergraphrag.utils import LLMIncompleteError, LLMRefusalError, LLMTruncationError

from revision.transport import (
    EMBED_DIGEST, EMBED_MODEL, ITEM, Transport as EmbeddingTransport,
    append_jsonl, canonical, file_hash, signature, write_json,
)

MODEL = "openai/gpt-4o-mini"
RETURNED_MODELS = {MODEL, "openai/gpt-4o-mini-2024-07-18"}
BASE_URL = "https://openrouter.ai/api/v1"
PROVIDER = "openrouter"
REASONING_EFFORT = "not_applicable"
TEMPERATURE = 0
PROXY = os.environ.get("OPENROUTER_PROXY") or None


class ProviderError(RuntimeError):
    pass


class CacheMissError(ProviderError):
    """An exact text request is unavailable during an offline recovery."""


def transient_error_code(code):
    """Retry explicit transport/provider availability errors, never policy errors."""
    if isinstance(code, str) and code.isdigit():
        code = int(code)
    return isinstance(code, int) and (
        code in (408, 409, 425, 429) or 500 <= code <= 599)


def normalize_usage(raw):
    """Keep comparable counters and retain the provider's full original usage."""
    prompt_details = raw.get("prompt_tokens_details") or {}
    completion_details = raw.get("completion_tokens_details") or {}
    return {
        "input_tokens": raw.get("prompt_tokens", 0),
        "output_tokens": raw.get("completion_tokens", 0),
        "total_tokens": raw.get("total_tokens", 0),
        "input_tokens_details": {
            "cached_tokens": prompt_details.get("cached_tokens", 0)},
        "output_tokens_details": {
            "reasoning_tokens": completion_details.get("reasoning_tokens", 0)},
    }


class Transport(EmbeddingTransport):
    def __init__(self, cache: Path, log_root: Path, workers=1, *, cache_only=False):
        # Do not call the legacy constructor: it reads the Fucheers credential.
        self.cache, self.log_root = Path(cache), Path(log_root)
        (self.cache / "responses").mkdir(parents=True, exist_ok=True)
        self.cache_only = cache_only
        self._key = ""
        if not cache_only:
            if os.environ.get("SHGRAPHRAG_ALLOW_MODEL_API") != "1":
                raise ProviderError("explicit_model_api_opt_in_required")
            self._key = os.environ.get("OPENROUTER_API_KEY", "").strip()
            if not self._key:
                raise ProviderError("missing_user_supplied_OPENROUTER_API_KEY")
            if os.environ.get("OPENROUTER_BASE_URL", "").rstrip("/") != BASE_URL:
                raise ProviderError("OpenRouter_origin_mismatch")
            if os.environ.get("OPENROUTER_MODEL") != MODEL:
                raise ProviderError("OpenRouter_model_mismatch")
            if (os.environ.get("OPENROUTER_PROXY") or None) != PROXY:
                raise ProviderError("OpenRouter_proxy_changed_after_import")
        capacity = int(os.environ.get("OPENROUTER_MAX_CONCURRENCY", "32"))
        if not 1 <= workers <= min(32, capacity):
            raise ProviderError("OpenRouter_concurrency_exceeds_shared_setting")
        self.client = LocalEmbeddingClient(self.log_root)
        self.text_client = None if cache_only else ProxyTextClient(PROXY, workers)
        self.semaphore = asyncio.Semaphore(workers)
        self.embed_semaphore = asyncio.Semaphore(1)
        self.locks = {}
        self.db = sqlite3.connect(self.cache / "embeddings.sqlite")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS vectors "
                        "(key TEXT PRIMARY KEY, vector BLOB NOT NULL)")
        self.calls = self.hits = self.embedding_calls = 0
        self.stages = Counter()
        self.item_counts = Counter()
        self.extracted_chunks = defaultdict(set)
        self.refused_chunks = defaultdict(set)
        self.truncated_chunks = defaultdict(set)
        self.incomplete_chunks = defaultdict(set)
        self.filtered_calls = self.filtered_cache_hits = 0
        self.truncated_calls = self.truncated_cache_hits = 0
        self.incomplete_calls = self.incomplete_cache_hits = 0
        self.active_calls = self.peak_active_calls = 0

    def raise_unavailable(self, saved, context, stage, cached=False):
        if saved["status"] == "output_truncated":
            if cached:
                self.truncated_cache_hits += 1
            else:
                self.truncated_calls += 1
            category, error_class = "truncated", LLMTruncationError
            chunks = self.truncated_chunks
        elif saved["status"] == "output_incomplete":
            if cached:
                self.incomplete_cache_hits += 1
            else:
                self.incomplete_calls += 1
            category, error_class = "incomplete", LLMIncompleteError
            chunks = self.incomplete_chunks
        elif saved["status"] == "provider_filtered":
            if cached:
                self.filtered_cache_hits += 1
            else:
                self.filtered_calls += 1
            category, error_class = "filtered", LLMRefusalError
            chunks = self.refused_chunks
        else:
            raise ProviderError("unknown_unavailable_output_status")
        self.stages[stage + "_" + category + ("_cache_hits" if cached else "")] += 1
        self.item_counts[(context.get("dataset"), context.get("item_id"), category)] += 1
        if stage == "extract" and context.get("chunk_id"):
            chunks[context.get("dataset")].add(context["chunk_id"])
        raise error_class(
            saved["error"], saved["request_sha256"], saved.get("text") or "",
            saved.get("response_id"))

    async def close(self):
        if self.text_client is not None:
            await self.text_client.aclose()
        await super().close()

    async def check_embedding(self):
        contract = await super().check_embedding()
        response = await self.client.get("http://127.0.0.1:11434/api/version")
        response.raise_for_status()
        contract["ollama_version"] = response.json().get("version")
        contract["endpoint"] = ENDPOINT
        contract["proxy"] = "bypassed: direct loopback client with trust_env=False"
        contract["http_recovery"] = {
            "max_attempts_per_exact_request": 4,
            "retry_statuses": [400, 408, 409, 425, 429, "5xx"],
            "retry_transport_errors": True,
            "exact_payload_unchanged_between_retries": True,
            "explicit_context_overflow": "separate_audited_truncate_true_fallback",
            "attempt_log": "embedding_http_attempts.jsonl",
            "raw_http_failure_artifacts": "cache/embedding_failures/",
        }
        contract["execution_options"] = dict(EMBED_OPTIONS)
        contract["overflow_policy"] = dict(OVERFLOW_POLICY)
        contract["runtime_scope"] = "dedicated_local_Ollama_process_port11436"
        service = Path(__file__).resolve().parents[1] / "monitoring/embedding_service_11436/service.json"
        if not service.exists():
            raise ProviderError("isolated_embedding_service_manifest_missing")
        record = json.loads(service.read_text(encoding="utf-8"))
        if record.get("endpoint") != ENDPOINT:
            raise ProviderError("isolated_embedding_service_endpoint_mismatch")
        write_json(self.log_root / "embedding_service.json", record)
        contract["service_manifest_sha256"] = file_hash(
            self.log_root / "embedding_service.json")
        return contract

    async def embed(self, texts):
        started, started_unix = time.perf_counter(), time.time()
        context = dict(ITEM.get())
        keys = {signature([EMBED_MODEL, EMBED_DIGEST, text]) for text in texts}
        missing = sum(
            self.db.execute("SELECT 1 FROM vectors WHERE key=?", (key,)).fetchone() is None
            for key in keys)
        before = self.embedding_calls
        vectors = await super().embed(texts)
        batches = self.embedding_calls - before
        self.item_counts[(context.get("dataset"), context.get("item_id"), "embedding_requests")] += batches
        append_jsonl(self.log_root / "embedding_invocations.jsonl", {
            **context, "started_unix": started_unix,
            "elapsed_s": time.perf_counter() - started,
            "texts": len(texts), "new_unique_texts": missing,
            "cached_or_repeated_texts": len(texts) - missing,
            "new_batch_requests": batches, "model": EMBED_MODEL,
            "digest": EMBED_DIGEST, "dimensions": 1024,
        })
        return vectors

    async def complete(self, prompt, system_prompt=None, history_messages=None,
                       max_tokens=None, **kwargs):
        if kwargs.get("stream"):
            raise ValueError("Benchmark requires complete non-streaming responses")
        context = dict(ITEM.get())
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
            initial = history_messages[0]["content"] if history_messages else prompt
            marker = "-Real Data-\n######################\nText: "
            tail = "\n######################\nOutput:"
            if marker in initial and tail in initial:
                import hashlib
                content = initial.split(marker, 1)[1].rsplit(tail, 1)[0]
                context["chunk_id"] = "chunk-" + hashlib.md5(content.encode()).hexdigest()
        cap = max_tokens or {
            "extract": 8192, "glean": 8192, "coverage": 64,
            "query_extract": 8192, "answer": 512, "summarize": 500,
        }.get(stage, 4096)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history_messages or [])
        messages.append({"role": "user", "content": prompt})
        body = {
            "model": MODEL, "messages": messages, "max_tokens": cap,
            "temperature": TEMPERATURE, "stream": False,
            "provider": {"require_parameters": True},
        }
        if stage == "answer":
            body["response_format"] = {"type": "json_object"}
        key = signature(body)
        path = self.cache / "responses" / (key + ".json")
        async with self.locks.setdefault(key, asyncio.Lock()):
            if path.exists():
                saved = json.loads(path.read_text(encoding="utf-8"))
                if (saved["request_sha256"] != key or saved["model"] != MODEL
                        or saved.get("provider") != PROVIDER
                        or saved.get("returned_model") not in RETURNED_MODELS
                        or saved.get("status") not in (
                            "completed", "provider_filtered", "output_truncated",
                            "output_incomplete")
                        or signature(saved["request"]) != key):
                    raise ProviderError("response_cache_identity_mismatch")
                self.hits += 1
                self.item_counts[(context.get("dataset"), context.get("item_id"), "hits")] += 1
                self.stages[stage + "_cache_hits"] += 1
                if saved["status"] == "completed" and stage == "extract" and context.get("chunk_id"):
                    self.extracted_chunks[context.get("dataset")].add(context["chunk_id"])
                append_jsonl(self.log_root / "cache_hits.jsonl", {
                    **context, "stage": stage, "request_sha256": key,
                    "time": time.time(), "provider": PROVIDER, "model": MODEL,
                    "status": saved["status"]})
                if saved["status"] != "completed":
                    self.raise_unavailable(saved, context, stage, cached=True)
                return saved["text"]
            if self.cache_only:
                relative = "cache/missing_text_requests/" + key + ".json"
                write_json(self.log_root / relative, {
                    "request_sha256": key, "request": body,
                    "context": context, "stage": stage,
                    "status": "cache_miss_no_request_submitted",
                })
                append_jsonl(self.log_root / "cache_misses.jsonl", {
                    **context, "stage": stage, "request_sha256": key,
                    "request_artifact": relative, "time": time.time(),
                    "network_requests": 0,
                })
                raise CacheMissError("offline_exact_text_cache_miss:" + stage + ":" + key)
            queued_at = time.perf_counter()
            request_invocation_id = key + "_" + str(time.time_ns())
            async with self.semaphore:
                admission_wait_s = time.perf_counter() - queued_at
                for attempt in range(1, 5):
                    start = time.perf_counter()
                    record = {
                        **context, "stage": stage, "attempt": attempt,
                        "request_sha256": key, "started_unix": time.time(),
                        "request_invocation_id": request_invocation_id,
                        "max_attempts": 4,
                        "model": MODEL, "provider": PROVIDER,
                        "reasoning_effort": REASONING_EFFORT,
                        "admission_wait_s": admission_wait_s if attempt == 1 else 0,
                    }
                    retryable = False
                    retry_delay = min(30, 2 ** attempt)
                    try:
                        self.active_calls += 1
                        self.peak_active_calls = max(self.peak_active_calls, self.active_calls)
                        try:
                            response = await self.text_client.post(
                                BASE_URL + "/chat/completions", json=body,
                                headers={"Authorization": "Bearer " + self._key})
                        finally:
                            self.active_calls -= 1
                        record["http_status"] = response.status_code
                        if response.status_code != 200:
                            retryable = transient_error_code(response.status_code)
                            relative = ("cache/text_failures/" + request_invocation_id
                                        + "_" + str(attempt) + ".json")
                            try:
                                raw = {"response": response.json()}
                            except ValueError:
                                raw = {"response_text": response.text}
                            write_json(self.log_root / relative, {
                                "request": body, "request_sha256": key,
                                "http_status": response.status_code, **raw,
                                "started_unix": record["started_unix"]})
                            record["failure_artifact"] = relative
                            delay = response.headers.get("retry-after", "")
                            if delay.isdigit():
                                retry_delay = min(60, max(retry_delay, int(delay)))
                            raise ProviderError("http_" + str(response.status_code))
                        data = response.json()
                        record["usage_raw"] = data.get("usage") or {}
                        record["usage"] = normalize_usage(record["usage_raw"])
                        if data.get("error"):
                            code = data["error"].get("code")
                            retryable = transient_error_code(code)
                            relative = ("cache/text_failures/" + request_invocation_id
                                        + "_" + str(attempt) + ".json")
                            write_json(self.log_root / relative, {
                                "request": body, "request_sha256": key,
                                "http_status": response.status_code, "response": data,
                                "started_unix": record["started_unix"]})
                            record.update(failure_artifact=relative,
                                          provider_error=data["error"],
                                          response_id=data.get("id"))
                            raise ProviderError("http_" + str(code))
                        record["returned_model"] = data.get("model")
                        record["upstream_provider"] = data.get("provider")
                        if data.get("model") not in RETURNED_MODELS:
                            raise ProviderError("returned_model_mismatch")
                        choices = data.get("choices") or []
                        if len(choices) != 1:
                            raise ProviderError("invalid_completion_choices")
                        choice = choices[0]
                        finish = choice.get("finish_reason")
                        record["finish_reason"] = finish
                        message = choice.get("message") or {}
                        if (finish in ("content_filter", "length") or finish is None
                                or message.get("refusal")):
                            if message.get("refusal") or finish == "content_filter":
                                status = "provider_filtered"
                                error = ("completion_content_filter" if finish == "content_filter"
                                         else "completion_refusal")
                            elif finish == "length":
                                status, error = "output_truncated", "completion_length"
                            else:
                                status, error = "output_incomplete", "completion_missing_finish_reason"
                                write_json(self.cache / "incomplete" / (
                                    key + "_" + str(time.time_ns()) + ".json"), {
                                        "request": body, "response": data})
                            saved = {
                                "request_sha256": key, "request": body, "model": MODEL,
                                "provider": PROVIDER, "returned_model": data["model"],
                                "upstream_provider": data.get("provider"),
                                "status": status, "error": error,
                                "text": message.get("content") or "",
                                "usage": record["usage"], "usage_raw": record["usage_raw"],
                                "response_id": data.get("id"), "finish_reason": finish,
                                "refusal": message.get("refusal"), "raw_response": data,
                                "elapsed_s": time.perf_counter() - start,
                            }
                            write_json(path, saved)
                            append_jsonl(self.log_root / "api_calls.jsonl", {
                                **record, "status": status, "error": error,
                                "elapsed_s": saved["elapsed_s"],
                                "response_id": saved["response_id"]})
                            self.raise_unavailable(saved, context, stage)
                        if choice.get("error"):
                            provider_error = choice["error"]
                            code = provider_error.get("code")
                            retryable = transient_error_code(code)
                            relative = ("cache/text_failures/" + request_invocation_id
                                        + "_" + str(attempt) + ".json")
                            write_json(self.log_root / relative, {
                                "request": body, "request_sha256": key,
                                "http_status": response.status_code, "response": data,
                                "started_unix": record["started_unix"]})
                            record.update(
                                failure_artifact=relative, provider_error=provider_error,
                                response_id=data.get("id"))
                            raise ProviderError("http_" + str(code))
                        if finish != "stop":
                            write_json(self.cache / "incomplete" / (
                                key + "_" + str(time.time_ns()) + ".json"), {
                                    "request": body, "response": data})
                            raise ProviderError("completion_" + str(finish))
                        text = choice.get("message", {}).get("content")
                        if not isinstance(text, str) or not text.strip():
                            retryable = True
                            raise ProviderError("empty_output_text")
                        saved = {
                            "request_sha256": key, "request": body, "model": MODEL,
                            "provider": PROVIDER, "returned_model": data["model"],
                            "upstream_provider": data.get("provider"),
                            "status": "completed", "text": text,
                            "usage": record["usage"], "usage_raw": record["usage_raw"],
                            "response_id": data.get("id"), "finish_reason": finish,
                            "system_fingerprint": data.get("system_fingerprint"),
                            "elapsed_s": time.perf_counter() - start,
                        }
                        write_json(path, saved)
                        record.update(
                            status="completed", elapsed_s=saved["elapsed_s"],
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
                    record.update(
                        status="failed", error=error, elapsed_s=time.perf_counter() - start,
                        retry_scheduled=retryable and attempt < 4,
                        retry_delay_s=retry_delay if retryable and attempt < 4 else 0)
                    append_jsonl(self.log_root / "api_calls.jsonl", record)
                    if not retryable or attempt == 4:
                        raise ProviderError(error)
                    await asyncio.sleep(retry_delay)
        raise AssertionError("unreachable")
