"""OpenRouter Chat Completions transport; credentials never enter artifacts."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import threading
import time
from typing import Any

import requests


BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "openai/gpt-4o-mini"
EFFORT = "none"
PROVIDER = "openrouter"
RUNTIME_DIRECTORY = ".runtime/4omini-reproduction"
RESPONSE_MODELS = {MODEL, "openai/gpt-4o-mini-2024-07-18",
                   "gpt-4o-mini", "gpt-4o-mini-2024-07-18"}
_IO_LOCK = threading.Lock()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def signature(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{threading.get_ident()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    # Windows readers/antivirus can briefly deny DELETE access to the target.
    # Keep atomic replacement and a finite budget; persistent failures surface.
    for attempt in range(8):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(min(0.5, 0.025 * 2 ** attempt))


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _IO_LOCK, path.open("a", encoding="utf-8") as stream:
        stream.write(canonical(value) + "\n")


class ProviderError(RuntimeError):
    def __init__(self, category: str, *, retryable: bool = False):
        self.category = category
        self.retryable = retryable
        super().__init__(category)


def request_body(system: str, prompt: str, max_tokens: int,
                 response_schema: dict | None = None) -> dict:
    # GPT-4o-mini has no configurable reasoning budget. Do not send the
    # Responses-only reasoning.effort or max_output_tokens fields.
    body = {"model": MODEL, "messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0, "stream": False,
        "provider": {"require_parameters": True}}
    if response_schema is not None:
        body["response_format"] = {
            "type": "json_schema", "json_schema": {
                "name": "research_output", "strict": True,
                "schema": response_schema}}
    return body


def normalized_usage(usage: dict) -> dict:
    return {
        **usage,
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "output_tokens_details": usage.get("completion_tokens_details") or {},
    }


class ChatCompletionsClient:
    def __init__(self, root: Path, *, concurrency: int = 32,
                 timeout: float = 240, attempts: int = 5,
                 cache_root: Path | None = None):
        self.root = root
        self.cache = (cache_root or root / "cache") / "chat_completions"
        self.log = root / "api_calls.jsonl"
        self.timeout = timeout
        self.attempts = attempts
        self.semaphore = threading.BoundedSemaphore(concurrency)
        self.concurrency = concurrency
        self._active = 0
        self._active_lock = threading.Lock()
        self._local = threading.local()
        if os.environ.get("SHGRAPHRAG_ALLOW_MODEL_API") != "1":
            raise ProviderError("explicit_model_api_opt_in_required")
        self._key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not self._key:
            raise ProviderError("missing_user_supplied_OPENROUTER_API_KEY")
        if os.environ.get("OPENROUTER_BASE_URL", BASE_URL).rstrip("/") != BASE_URL:
            raise ProviderError("unexpected_openrouter_origin")
        if os.environ.get("OPENROUTER_MODEL", MODEL) != MODEL:
            raise ProviderError("unexpected_openrouter_model")
        self.proxy = os.environ.get("OPENROUTER_PROXY") or None

    def session(self) -> requests.Session:
        if not hasattr(self._local, "session"):
            session = requests.Session()
            session.trust_env = False
            # This session calls only the fixed external origin. Embeddings
            # have a separate direct session, so loopback never uses Clash.
            session.proxies = {"https": self.proxy}
            self._local.session = session
        return self._local.session

    def call(self, *, stage: str, item_id: str, system: str, prompt: str,
             max_tokens: int = 4096, response_schema: dict | None = None) -> dict:
        body = request_body(system, prompt, max_tokens, response_schema)
        cache_key = signature(body)
        path = self.cache / cache_key[:2] / f"{cache_key}.json"
        if path.exists():
            result = json.loads(path.read_text(encoding="utf-8"))
            if (result.get("request_sha256") != cache_key
                    or result.get("model") != MODEL
                    or result.get("provider") != PROVIDER):
                raise ProviderError("cache_hash_mismatch")
            append_jsonl(self.root / "cache_hits.jsonl", {
                "stage": stage, "item_id": item_id, "request_sha256": cache_key})
            return {**result, "cache_hit": True}
        queued_at = time.perf_counter()
        with self.semaphore:
            queue_wait = time.perf_counter() - queued_at
            for attempt in range(1, self.attempts + 1):
                started = time.perf_counter()
                record = {"stage": stage, "item_id": item_id,
                          "request_sha256": cache_key, "model": MODEL,
                          "provider": PROVIDER, "reasoning_effort": EFFORT,
                          "attempt": attempt,
                          "queue_wait_s": queue_wait if attempt == 1 else 0,
                          "configured_concurrency": self.concurrency,
                          "started_unix": time.time()}
                retry_delay = min(30, 2 ** attempt) + random.random()
                try:
                    with self._active_lock:
                        self._active += 1
                        record["active_http_at_dispatch"] = self._active
                    try:
                        response = self.session().post(
                            BASE_URL + "/chat/completions", json=body,
                            headers={"Authorization": "Bearer " + self._key},
                            timeout=(15, self.timeout), allow_redirects=False)
                    finally:
                        with self._active_lock:
                            self._active -= 1
                    record["http_status"] = response.status_code
                    if response.status_code != 200:
                        retry_after = response.headers.get("Retry-After", "")
                        if retry_after.isdigit():
                            retry_delay = min(60, max(retry_delay, int(retry_after)))
                        category = f"http_{response.status_code}"
                        raise ProviderError(category, retryable=(
                            response.status_code in (408, 409, 425, 429)
                            or response.status_code >= 500))
                    data = response.json()
                    record["usage"] = normalized_usage(data.get("usage") or {})
                    record["response_model"] = data.get("model")
                    record["upstream_provider"] = data.get("provider")
                    record["response_id"] = data.get("id")
                    if data.get("error"):
                        raise ProviderError("embedded_provider_error", retryable=True)
                    if data.get("model") not in RESPONSE_MODELS:
                        raise ProviderError("response_model_mismatch")
                    choice = data["choices"][0]
                    message = choice["message"]
                    record["finish_reason"] = choice.get("finish_reason")
                    record["native_finish_reason"] = choice.get("native_finish_reason")
                    if message.get("refusal"):
                        raise ProviderError("model_refusal")
                    if not isinstance(choice.get("finish_reason"), str):
                        raise ProviderError("invalid_finish_reason", retryable=True)
                    if choice.get("finish_reason") != "stop":
                        reason = choice.get("finish_reason")
                        raise ProviderError("finish_" + str(reason),
                                            retryable=reason in ("content_filter", "error"))
                    if (message.get("reasoning") or message.get("reasoning_content")
                            or record["usage"]["output_tokens_details"].get(
                                "reasoning_tokens", 0)):
                        raise ProviderError("unexpected_reasoning_output")
                    text = message.get("content") or ""
                    if not text.strip():
                        raise ProviderError("empty_output_text", retryable=True)
                    elapsed = time.perf_counter() - started
                    result = {"request_sha256": cache_key, "model": MODEL,
                              "provider": PROVIDER,
                              "response_model": data["model"],
                              "upstream_provider": data.get("provider"),
                              "status": "completed", "reasoning_effort": EFFORT,
                              "reasoning_configuration": "non_reasoning_model_no_parameter",
                              "text": text, "usage": record["usage"],
                              "response_id": data.get("id"), "elapsed_s": elapsed,
                              "queue_wait_s": queue_wait,
                              "stage": stage, "item_id": item_id}
                    record.update(status="completed", elapsed_s=elapsed,
                                  usage=result["usage"],
                                  response_id=result["response_id"])
                    write_json(path, result)
                    append_jsonl(self.log, record)
                    return {**result, "cache_hit": False}
                except requests.exceptions.ChunkedEncodingError:
                    record["transport_error_type"] = "ChunkedEncodingError"
                    error = ProviderError("network_incomplete_response", retryable=True)
                except (requests.Timeout, requests.ConnectionError):
                    error = ProviderError("network_timeout_or_connection",
                                          retryable=True)
                except (ValueError, KeyError, IndexError, TypeError):
                    error = ProviderError("invalid_response_json", retryable=True)
                except ProviderError as caught:
                    error = caught
                record.update(status="failed", error=error.category,
                              elapsed_s=time.perf_counter() - started)
                append_jsonl(self.log, record)
                if not error.retryable or attempt == self.attempts:
                    raise error
                time.sleep(retry_delay)
        raise AssertionError("unreachable")


# Stable internal interface for the already verified research pipeline.
ResponsesClient = ChatCompletionsClient


def parse_json(text: str) -> Any:
    """Decode one complete JSON value, tolerating surrounding Markdown only."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(stripped)
