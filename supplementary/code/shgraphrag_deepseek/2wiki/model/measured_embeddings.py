"""Measure and bound concurrent local Ollama calls without changing vectors."""

import hashlib
import time
import uuid

from langchain_core.embeddings import Embeddings
from ollama import ResponseError

from config.settings import EMBEDDING_MAX_IN_FLIGHT
from reproduction.runtime import (
    ROOT, RequestSlots, append_jsonl, check_cancelled, request_context, run_root, utc_now,
)


class MeasuredEmbeddings(Embeddings):
    def __init__(self, delegate):
        self.delegate = delegate
        # All wrappers and processes in this experiment share the same slots.
        # The text API uses its own separately configured gate.
        self.slots = RequestSlots(
            ROOT / ".runtime/bge_request_slots", capacity=EMBEDDING_MAX_IN_FLIGHT
        )

    def _measure(self, texts, operation):
        logical_call_id = uuid.uuid4().hex
        base = {
            **request_context(), "logical_call_id": logical_call_id, "operation": operation,
            "model": self.delegate.model, "endpoint": self.delegate.base_url,
            "items": len(texts),
            "text_sha256": [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts],
            "characters": sum(len(text) for text in texts),
            "provider_token_usage": None,
            "token_usage_note": "Original LangChain Ollama wrapper exposes vectors, not response usage",
        }
        with self.slots.acquire(cancellation_check=check_cancelled) as slot:
            for attempt in range(1, 4):
                check_cancelled()
                started = time.perf_counter()
                record = {
                    **base, "attempt": attempt, "started_at": utc_now(),
                    "sent_unix": time.time(),
                    "local_gate_capacity": self.slots.capacity,
                    "local_gate_slot": slot["slot"],
                    "queue_seconds": slot["queue_seconds"] if attempt == 1 else 0.0,
                }
                try:
                    if operation == "embed_query":
                        result = self.delegate.embed_query(texts[0])
                    else:
                        result = self.delegate.embed_documents(texts)
                    record["status"] = "completed"
                    return result
                except Exception as error:
                    retryable = (
                        isinstance(error, ResponseError)
                        and error.status_code == 400
                        and "127.0.0.1:" in str(error)
                        and "/tokenize" in str(error)
                        and (
                            "refused" in str(error).lower()
                            or "lacked sufficient buffer space" in str(error).lower()
                        )
                    )
                    record.update(
                        status="failed", error_type=type(error).__name__, error=str(error),
                        retryable_local_runner_refusal=retryable,
                    )
                    if not retryable or attempt == 3:
                        raise
                finally:
                    record["elapsed_seconds"] = time.perf_counter() - started
                    record["received_unix"] = time.time()
                    record["finished_at"] = utc_now()
                    append_jsonl(run_root() / "embedding_calls.jsonl", record)
                time.sleep(2 ** (attempt - 1))

    def embed_documents(self, texts):
        if not texts:
            return []
        # The original indexers already use a maximum subbatch of 32.
        # Bound the large per-document entity batches the same way, preserving
        # every input and output position while limiting Ollama runner fan-out.
        vectors = []
        for offset in range(0, len(texts), 32):
            vectors.extend(self._measure(texts[offset:offset + 32], "embed_documents"))
        return vectors

    def embed_query(self, text):
        return self._measure([text], "embed_query")
