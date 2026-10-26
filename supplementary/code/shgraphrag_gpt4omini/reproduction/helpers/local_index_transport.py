"""Measured local index transport recovery, without changing embedding payloads."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import time

from reproduction.client import append_jsonl, signature
from reproduction.embedding import ENDPOINT


def socket_resource_error(response):
    if response.status_code != 400:
        return False
    try:
        message = response.json().get("error", "").lower()
    except (ValueError, AttributeError, TypeError):
        return False
    return (
        "127.0.0.1:" in message
        and ("dial tcp" in message or "connectex" in message)
        and (
            "insufficient buffer space" in message
            or "lacked sufficient buffer space" in message
            or "only one usage of each socket address" in message
        )
    )


def recovering_post(operation, log, *, interval=2.0, max_attempts=4,
                    clock=time.perf_counter, sleeper=time.sleep,
                    wall_clock=time.time):
    if interval < 2 or not 1 <= max_attempts <= 4:
        raise ValueError("Expected index pacing >=2s and at most four attempts")
    last_start = None
    cooldowns = (10.0, 30.0, 60.0)

    def post(url, **kwargs):
        nonlocal last_start
        if url != ENDPOINT + "/api/embed" or set(kwargs) != {"json", "timeout"}:
            raise ValueError("Recovery is restricted to the ordinary local embed request")
        payload = kwargs["json"]
        request_hash = signature(payload)
        for attempt in range(max_attempts):
            if last_start is not None:
                delay = last_start + interval - clock()
                if delay > 0:
                    sleeper(delay)
            last_start = clock()
            started_unix = wall_clock()
            response = operation(url, **kwargs)
            retryable = socket_resource_error(response)
            retry = retryable and attempt + 1 < max_attempts
            record = {
                "stage": "index_local_transport",
                "started_unix": started_unix, "elapsed_s": clock() - last_start,
                "request_sha256": request_hash, "attempt": attempt + 1,
                "max_attempts": max_attempts, "http_status": response.status_code,
                "input_rows": len(payload["input"]), "minimum_post_interval_s": interval,
                "socket_resource_error": retryable, "will_retry": retry,
                "cooldown_s": cooldowns[attempt] if retry else 0,
                "payload_changed": False,
            }
            if retryable:
                record["local_service_error_body"] = response.text[:4000]
            append_jsonl(log, record)
            if not retry:
                return response
            response.close()
            sleeper(cooldowns[attempt])
        raise AssertionError("Finite local transport attempt loop escaped")

    return post


@contextmanager
def index_transport(embedding, log: Path, *, interval=2.0, max_attempts=4):
    original = embedding.session.post
    embedding.session.post = recovering_post(
        original, log, interval=interval, max_attempts=max_attempts)
    try:
        yield
    finally:
        embedding.session.post = original
