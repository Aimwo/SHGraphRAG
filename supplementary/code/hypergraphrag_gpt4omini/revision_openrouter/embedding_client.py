"""Direct loopback embedding HTTP with exact-body retries and raw diagnostics."""
from __future__ import annotations

import asyncio
import json as json_module
from pathlib import Path
import time

import httpx

from revision.transport import (
    EMBED_DIGEST, EMBED_MODEL, ITEM, append_jsonl, file_hash, signature,
    write_json,
)

LOCAL_BASE_URL = "http://127.0.0.1:11436"
LEGACY_BASE_URL = "http://127.0.0.1:11434"
ENDPOINT = LOCAL_BASE_URL + "/api/embed"
MAX_ATTEMPTS = 4
RETRY_STATUS = {400, 408, 409, 425, 429}
EMBED_OPTIONS = {"num_ctx": 8192, "num_batch": 8192}
OVERFLOW_ERROR = "the input length exceeds the context length"
OVERFLOW_POLICY = {
    "trigger": "HTTP400_exact_input_length_exceeds_context_error",
    "primary_truncate": False,
    "fallback_truncate": True,
    "fallback": "Ollama_native_context_truncation_same_model_text_and_options",
    "scope": "all_embedding_inputs_without_gold_or_dataset_specific_selection",
    "raw_provenance": "cache/embedding_overflows/",
    "in_range_cached_vectors": "retain_prior_successful_truncate_false_vectors",
}


class LocalEmbeddingClient:
    def __init__(self, log_root: Path, *, transport=None):
        self.log_root = Path(log_root)
        self.http = httpx.AsyncClient(
            trust_env=False, follow_redirects=False,
            timeout=httpx.Timeout(240, connect=20), transport=transport)

    async def get(self, url):
        # The inherited vector/cache implementation supplies its legacy origin;
        # route only the explicitly allowed local model-inspection paths.
        path = next((url[len(base):] for base in (LOCAL_BASE_URL, LEGACY_BASE_URL)
                     if url.startswith(base + "/api/")), None)
        if path not in ("/api/tags", "/api/ps", "/api/version"):
            raise ValueError("Embedding client only permits the pinned loopback origin")
        return await self.http.get(LOCAL_BASE_URL + path)

    async def aclose(self):
        await self.http.aclose()

    async def post(self, url, *, json):
        if (url not in (ENDPOINT, LEGACY_BASE_URL + "/api/embed")
                or json.get("model") != EMBED_MODEL
                or json.get("truncate") is not False
                or json.get("options", EMBED_OPTIONS) != EMBED_OPTIONS
                or not isinstance(json.get("input"), list)
                or any(not isinstance(text, str) for text in json["input"])):
            raise ValueError("Embedding request identity mismatch")
        json = {**json, "options": dict(EMBED_OPTIONS)}
        return await self._post(json)

    async def _post(self, json, overflow_artifact=None):
        request_hash = signature(json)
        invocation_id = f"{request_hash}_{time.time_ns()}"
        context = dict(ITEM.get())
        for attempt in range(1, MAX_ATTEMPTS + 1):
            started, started_unix = time.perf_counter(), time.time()
            record = {
                **context, "request_sha256": request_hash,
                "invocation_id": invocation_id,
                "attempt": attempt, "max_attempts": MAX_ATTEMPTS,
                "started_unix": started_unix,
                "endpoint": ENDPOINT, "model": EMBED_MODEL,
                "digest": EMBED_DIGEST, "truncate": json["truncate"],
                "execution_options": dict(EMBED_OPTIONS),
                "texts": len(json["input"]),
                "input_char_lengths": [len(text) for text in json["input"]],
            }
            if overflow_artifact:
                record["overflow_artifact"] = overflow_artifact
            response = None
            caught = None
            retryable = False
            overflow = False
            try:
                response = await self.http.post(ENDPOINT, json=json)
                record["http_status"] = response.status_code
                if response.status_code == 200:
                    record["status"] = "completed"
                    if overflow_artifact:
                        try:
                            payload = response.json()
                        except ValueError:
                            payload = None
                        if payload is not None:
                            path = self.log_root / overflow_artifact
                            evidence = json_module.loads(
                                path.read_text(encoding="utf-8"))
                            evidence["response"] = payload
                            evidence["response_sha256"] = signature(payload)
                            evidence["completed_unix"] = time.time()
                            write_json(path, evidence)
                            record["response_sha256"] = evidence["response_sha256"]
                            record["overflow_artifact_sha256"] = file_hash(path)
                else:
                    record.update(status="failed", error="http_" + str(response.status_code))
                    retryable = (response.status_code in RETRY_STATUS
                                 or response.status_code >= 500)
                    try:
                        raw = {"response": response.json()}
                    except ValueError:
                        raw = {"response_text": response.text}
                    overflow = (
                        response.status_code == 400
                        and (raw.get("response") or {}).get("error") == OVERFLOW_ERROR)
                    if overflow:
                        retryable = False
                    relative = (
                        "cache/embedding_failures/"
                        f"{request_hash}_{time.time_ns()}_{attempt}.json")
                    write_json(self.log_root / relative, {
                        "request_sha256": request_hash, "request": json,
                        "http_status": response.status_code, **raw,
                        "started_unix": started_unix,
                    })
                    record["failure_artifact"] = relative
                    record["failure_artifact_sha256"] = file_hash(
                        self.log_root / relative)
            except httpx.TransportError as exc:
                caught = exc
                retryable = True
                record.update(status="failed", error=type(exc).__name__)
            record["elapsed_s"] = time.perf_counter() - started
            if overflow_artifact and "overflow_artifact_sha256" not in record:
                path = self.log_root / overflow_artifact
                if path.is_file():
                    record["overflow_artifact_sha256"] = file_hash(path)
            retry = retryable and attempt < MAX_ATTEMPTS
            record["retry_scheduled"] = retry
            record["retry_delay_s"] = min(30, 2 ** attempt) if retry else 0
            append_jsonl(self.log_root / "embedding_http_attempts.jsonl", record)
            if overflow and overflow_artifact is None:
                fallback = {**json, "truncate": True}
                relative = (
                    "cache/embedding_overflows/"
                    f"{request_hash}_{time.time_ns()}.json")
                write_json(self.log_root / relative, {
                    "policy": dict(OVERFLOW_POLICY),
                    "original_request": json,
                    "original_request_sha256": request_hash,
                    "original_failure_artifact": record["failure_artifact"],
                    "fallback_request": fallback,
                    "fallback_request_sha256": signature(fallback),
                    "created_unix": time.time(),
                    "diagnostic_only": False,
                })
                return await self._post(fallback, overflow_artifact=relative)
            if not retry:
                if caught is not None:
                    raise caught
                return response
            await asyncio.sleep(record["retry_delay_s"])
        raise AssertionError("Unreachable")
