"""Task-scoped transport; every physical attempt has its own saved evidence."""

import hashlib
import json
import os
import pickle
import time
import uuid
from typing import Any, Optional

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

from config.settings import TEXT_RECOVERABLE_ATTEMPTS
from reproduction.relay_config import config_identity, load_relay_config, read_authorized_key
from reproduction.runtime import (
    RequestSlots, append_jsonl, call_context, check_cancelled, request_context, run_root,
    utc_now, write_json,
)

RETRY_POLICY_ID = "user-approved-direct-retry-20261025"


class RecoverableRelayError(RuntimeError):
    def __init__(self, message, fault, request_sha256):
        super().__init__(message)
        self.fault = fault
        self.request_sha256 = request_sha256


def normalized_usage(usage):
    if not isinstance(usage, dict):
        return None
    details = usage.get("prompt_tokens_details") or {}
    output_details = usage.get("completion_tokens_details") or {}
    cached = details.get("cached_tokens", usage.get("prompt_cache_hit_tokens"))
    result = {
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "input_tokens_details": {"cached_tokens": cached} if cached is not None else {},
        "output_tokens_details": {
            "reasoning_tokens": output_details["reasoning_tokens"]
        } if "reasoning_tokens" in output_details else {},
    }
    return result


def response_text(data):
    choices = data.get("choices") or []
    if len(choices) != 1 or choices[0].get("finish_reason") != "stop":
        raise RuntimeError("Provider did not return one complete, normally stopped answer")
    text = (choices[0].get("message") or {}).get("content")
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("Provider returned no nonempty answer content")
    return text


def refusal_text(data):
    """Preserve an explicit provider refusal as an observed model outcome."""
    choices = data.get("choices") or []
    if len(choices) != 1 or choices[0].get("finish_reason") != "content_filter":
        raise RuntimeError("Expected one explicit provider content-filter outcome")
    text = (choices[0].get("message") or {}).get("content")
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("Provider refusal has no text to preserve")
    return text


def save_refusal_evidence(root, record, text, *, origin="live_response"):
    append_jsonl(root / "provider_refusals.jsonl", {
        "at": utc_now(), "call_id": record["id"], "stage": record.get("stage"),
        "chunk_hash": record.get("chunk_hash"), "question_id": record.get("question_id"),
        "source_files": record.get("source_files"),
        "http_status": record.get("http_status"), "finish_reason": "content_filter",
        "credential_id": record.get("credential_id"), "model": record.get("returned_model"),
        "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "content_chars": len(text), "origin": origin,
        "provider_route_epoch": record.get("provider_route_epoch"),
        "policy": "Preserve exact refusal text; no prompt rewrite or replacement; retain all sample IDs",
    })


def active_refusal_chunk_hashes(root):
    """Match active cache text to preserved refusals, retaining old attempts."""
    ledger = root / "provider_refusals.jsonl"
    if not ledger.exists():
        return set()
    expected = {}
    for line in ledger.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if entry.get("stage") == "graph_extraction" and entry.get("chunk_hash"):
            expected.setdefault(entry["chunk_hash"], set()).add(entry["content_sha256"])
    active = set()
    for key, hashes in expected.items():
        try:
            with (root / "cache/graph" / (key + ".pkl")).open("rb") as stream:
                text = pickle.load(stream)
            if hashlib.sha256(text.encode("utf-8")).hexdigest() in hashes:
                active.add(key)
        except (FileNotFoundError, EOFError, pickle.UnpicklingError):
            # A live writer may not yet have committed this cache entry.
            continue
    return active


class DeepSeekRelayChat(BaseChatModel):
    model_name: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    _client: Any = PrivateAttr(default=None)

    @property
    def _llm_type(self):
        return "isolated-deepseek-relay-chat-completions"

    @property
    def _identifying_params(self):
        config = load_relay_config()
        return {
            "model": self.model_name,
            "provider": config["provider"],
            "credential_id": config["credential_id"],
            "config_sha256": config_identity(config),
        }

    def _payload(self, messages, config):
        if self.model_name != config["model"]:
            raise ValueError("Model differs from the pinned relay contract")
        roles = {"human": "user", "ai": "assistant", "system": "system"}
        original = []
        for message in messages:
            if message.type not in roles or not isinstance(message.content, str):
                raise ValueError("Only the original text chat messages are supported")
            original.append({"role": roles[message.type], "content": message.content})
        extra = config.get("generation_parameters", {})
        if set(extra) - {"thinking", "reasoning_effort"}:
            raise ValueError("Unexpected model-specific generation override")
        return {"model": self.model_name, "messages": original, "stream": False, **extra}

    def _http_client(self, config):
        if self._client is None:
            self._client = httpx.Client(
                trust_env=False, follow_redirects=False,
                timeout=httpx.Timeout(config["request_timeout_seconds"], connect=30, pool=30),
                limits=httpx.Limits(
                    max_connections=config["max_in_flight"],
                    max_keepalive_connections=config["max_in_flight"],
                ),
            )
        return self._client

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        logical_id = uuid.uuid4().hex
        previous_call_id = None
        for attempt in range(1, TEXT_RECOVERABLE_ATTEMPTS + 1):
            with call_context(
                logical_call_id=logical_id, logical_attempt=attempt,
                retry_of_call_id=previous_call_id, retry_policy_id=RETRY_POLICY_ID,
            ):
                try:
                    return self._generate_once(
                        messages, stop=stop, run_manager=run_manager, **kwargs
                    )
                except RecoverableRelayError as error:
                    root = run_root()
                    exhausted = attempt == TEXT_RECOVERABLE_ATTEMPTS
                    append_jsonl(root / "provider_retry_events.jsonl", {
                        "at": utc_now(), "logical_call_id": logical_id,
                        "failed_call_id": error.fault["call_id"],
                        "logical_attempt": attempt,
                        "maximum_attempts": TEXT_RECOVERABLE_ATTEMPTS,
                        "request_sha256": error.request_sha256,
                        "http_status": error.fault.get("http_status"),
                        "retry_policy_id": RETRY_POLICY_ID,
                        "outcome": "exhausted" if exhausted else "retry_scheduled",
                        "retry_delay_seconds": 0,
                    })
                    if exhausted:
                        write_json(root / "FATAL_PROVIDER_ERROR.json", {
                            **error.fault,
                            "logical_call_id": logical_id,
                            "immediate_attempts_exhausted": attempt,
                            "retry_policy_id": RETRY_POLICY_ID,
                        })
                        raise
                    previous_call_id = error.fault["call_id"]

    def _generate_once(self, messages, stop=None, run_manager=None, **kwargs):
        if stop:
            raise ValueError("Original benchmark does not supply stop sequences")
        config = load_relay_config()
        root = run_root()
        check_cancelled()
        identity = config_identity(config)
        run_config = root / "effective_config.json"
        if not run_config.exists():
            raise RuntimeError("A saved effective run configuration is required before any request")
        effective = json.loads(run_config.read_text(encoding="utf-8"))
        if effective.get("relay_config_sha256") != identity:
            raise ValueError("Refusing to mix provider, credential or model configurations")
        payload = self._payload(messages, config)
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        request_hash = hashlib.sha256(encoded.encode()).hexdigest()
        call_id = uuid.uuid4().hex
        request_dir = root / "requests"
        endpoint = config["base_url"].rstrip("/") + "/chat/completions"
        record = {
            "id": call_id, **request_context(), "pid": os.getpid(),
            "provider": config["provider"], "credential_id": config["credential_id"],
            "relay_config_sha256": identity, "endpoint": endpoint,
            "requested_model": self.model_name, "started_at": utc_now(),
            "request_sha256": request_hash, "physical_attempt": 1,
            "api_request_started": False, "usage": None, "status": "waiting_for_slot",
        }
        route_path = root / "provider_route_epoch.json"
        if route_path.exists():
            record["provider_route_epoch"] = json.loads(
                route_path.read_text(encoding="utf-8")
            )["id"]
        started = time.perf_counter()
        write_json(request_dir / f"{call_id}.input.json", payload)
        write_json(request_dir / f"{call_id}.record.json", record)
        key = None
        try:
            with RequestSlots().acquire(check_cancelled) as slot:
                check_cancelled()
                # Decryption happens only for this actual authorized work item.
                key = read_authorized_key(config)
                record.update(slot)
                record.update(sent_at=utc_now(), sent_unix=time.time(),
                              api_request_started=True, status="in_flight")
                write_json(request_dir / f"{call_id}.record.json", record)
                network_started = time.perf_counter()
                response = self._http_client(config).post(
                    endpoint, json=payload,
                    headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
                )
                record.update(
                    response_seconds=time.perf_counter() - network_started,
                    received_unix=time.time(), http_status=response.status_code,
                )
                record["response_headers"] = {
                    name: response.headers[name] for name in (
                        "content-type", "x-request-id", "request-id",
                        "x-new-api-request-id", "retry-after",
                        "x-ratelimit-limit-requests", "x-ratelimit-remaining-requests",
                    ) if name in response.headers
                }
                safe_body = response.text.replace(key, "[REDACTED]")
                raw_path = request_dir / f"{call_id}.response.txt"
                raw_path.write_text(safe_body, encoding="utf-8")
                record["response_body_sha256"] = hashlib.sha256(safe_body.encode()).hexdigest()
                try:
                    data = json.loads(safe_body)
                except ValueError:
                    fault = {
                        "call_id": call_id, "http_status": response.status_code,
                        "error": "Non-JSON relay response", "at": utc_now(),
                        "retryable_after_cooldown": response.status_code in (429, 502, 503, 504),
                    }
                    if fault["retryable_after_cooldown"]:
                        raise RecoverableRelayError(
                            f"Relay HTTP {response.status_code}: non-JSON response",
                            fault, request_hash,
                        )
                    if response.status_code < 500 or response.status_code in (502, 503, 504):
                        write_json(root / "FATAL_PROVIDER_ERROR.json", fault)
                    raise RuntimeError(f"Relay HTTP {response.status_code}: non-JSON response")
                write_json(request_dir / f"{call_id}.response.json", data)
                usage = data.get("usage")
                record.update(
                    raw_usage=usage, usage=normalized_usage(usage),
                    returned_model=data.get("model"), response_id=data.get("id"),
                )
                choices = data.get("choices") or []
                record["finish_reason"] = choices[0].get("finish_reason") if choices else None
                provider_error = data.get("error")
                gateway_request_failed = (
                    response.status_code == 500
                    and isinstance(provider_error, dict)
                    and provider_error.get("type") == "new_api_error"
                    and provider_error.get("code") == "do_request_failed"
                )
                if response.status_code in (400, 401, 402, 403, 404, 422, 429, 502, 503, 504) or gateway_request_failed:
                    fault = {
                        "call_id": call_id, "http_status": response.status_code,
                        "error": data.get("error"), "at": utc_now(),
                        "retryable_after_cooldown": (
                            response.status_code in (429, 502, 503, 504) or gateway_request_failed
                        ),
                        **({"transport_error_kind": "gateway_do_request_failed"}
                           if gateway_request_failed else {}),
                    }
                    if fault["retryable_after_cooldown"]:
                        raise RecoverableRelayError(
                            f"DeepSeek relay HTTP {response.status_code}", fault, request_hash,
                        )
                    write_json(root / "FATAL_PROVIDER_ERROR.json", fault)
                if response.status_code != 200:
                    raise RuntimeError(f"DeepSeek relay HTTP {response.status_code}")
                if data.get("model") not in config["allowed_returned_models"]:
                    write_json(root / "FATAL_PROVIDER_ERROR.json", {
                        "call_id": call_id, "error": "Unexpected returned model", "at": utc_now(),
                    })
                    raise RuntimeError("Returned model is outside the pinned allowlist")
                if record["finish_reason"] == "content_filter":
                    text = refusal_text(data)
                    record["response_status"] = "refused"
                    save_refusal_evidence(root, record, text)
                else:
                    text = response_text(data)
                    record["response_status"] = "completed"
            record["status"] = record["response_status"]
        except Exception as error:
            record.update(status="failed", error_type=type(error).__name__,
                          error=str(error).replace(key, "[REDACTED]") if key else str(error))
            if isinstance(error, httpx.ConnectError) and "getaddrinfo failed" in str(error).lower():
                record["transport_error_kind"] = "dns_resolution"
                marker = root / "FATAL_PROVIDER_ERROR.json"
                if not marker.exists():
                    write_json(marker, {
                        "call_id": call_id, "http_status": None, "at": utc_now(),
                        "transport_error_kind": "dns_resolution",
                        "error": {"type": type(error).__name__, "message": record["error"]},
                        "retryable_after_cooldown": True,
                    })
            raise
        finally:
            record.update(finished_at=utc_now(), elapsed_seconds=time.perf_counter() - started)
            write_json(request_dir / f"{call_id}.record.json", record)
        metadata = {
            "model_name": record["returned_model"], "response_id": record["response_id"],
            "call_id": call_id, "credential_id": config["credential_id"],
            "provider_outcome": record["response_status"],
            "token_usage": record["usage"], "transport": "chat-completions",
        }
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=text, response_metadata=metadata))],
            llm_output=metadata,
        )
