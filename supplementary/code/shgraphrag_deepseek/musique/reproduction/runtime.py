"""Run-scoped bookkeeping and an OS-released, cross-process request limit."""

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
CALL_CONTEXT = ContextVar("shgraph_call_context", default={})
_journal_lock = threading.Lock()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def run_root():
    value = os.environ.get("SHGRAPH_RUN_ROOT")
    if not value:
        raise RuntimeError("SHGRAPH_RUN_ROOT must explicitly identify this experiment")
    path = Path(value).resolve()
    if not path.is_relative_to(ROOT / "reproduction/runs"):
        raise RuntimeError("Run output must stay inside this isolated checkout")
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _journal_lock, path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")
        stream.flush()


@contextmanager
def call_context(**values):
    token = CALL_CONTEXT.set({**CALL_CONTEXT.get(), **values})
    try:
        yield
    finally:
        CALL_CONTEXT.reset(token)


def request_context():
    return {"stage": os.environ.get("SHGRAPH_STAGE", "unspecified"), **CALL_CONTEXT.get()}


def completed_interval_peak(records):
    """Peak from recorded request intervals; unfinished calls are not inferred."""
    events = []
    for row in records:
        began, ended = row.get("sent_unix"), row.get("received_unix")
        if began is not None and ended is not None and ended > began:
            events.extend([(began, 1), (ended, -1)])
    active = peak = 0
    # Ending an interval before starting another at the same time avoids
    # counting touching, non-overlapping intervals as concurrent.
    for _, change in sorted(events):
        active += change
        peak = max(peak, active)
    return peak


def check_cancelled():
    root = run_root()
    for name in ("STOP_REQUESTED", "FATAL_PROVIDER_ERROR.json"):
        if (root / name).exists():
            raise RuntimeError(f"Experiment stopped: {name}")


class RequestSlots:
    """Byte locks release automatically if a thread's owning process exits.

    Every adapter/process for this checkout uses the same 64 slot files.
    The lock covers request transmission through complete response consumption.
    It never expires while a request is still being read.
    """

    def __init__(self, directory=None, capacity=64):
        if not 1 <= capacity <= 64:
            raise ValueError("Request capacity must be between 1 and 64")
        self.directory = Path(directory or ROOT / ".runtime/deepseek_relay_request_slots")
        self.capacity = capacity
        self.directory.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def acquire(self, cancellation_check=None):
        started = time.perf_counter()
        handle = None
        slot_number = None
        while handle is None:
            if cancellation_check:
                cancellation_check()
            for index in range(self.capacity):
                candidate = (self.directory / f"slot_{index:02d}.lock").open("a+b")
                # Windows byte locks may extend past EOF. Initializing the
                # byte before locking races with another wrapper's byte lock.
                candidate.seek(0)
                try:
                    if os.name == "nt":
                        import msvcrt
                        msvcrt.locking(candidate.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(candidate.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except (OSError, BlockingIOError):
                    candidate.close()
                    continue
                handle = candidate
                slot_number = index
                break
            if handle is None:
                time.sleep(0.05)
        try:
            yield {"slot": slot_number, "queue_seconds": time.perf_counter() - started}
        finally:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()
