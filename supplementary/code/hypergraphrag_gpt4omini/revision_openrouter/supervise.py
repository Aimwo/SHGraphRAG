"""Durable local worker supervision; no scheduler or notification service."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from .run import BASE, campaign_lock, read_json, re_safe_name
from .transport import append_jsonl, write_json


def retryable(failure):
    error_type, error = failure.get("error_type", ""), failure.get("error", "")
    if error_type in ("ReadTimeout", "ConnectTimeout", "WriteTimeout", "PoolTimeout",
                      "ConnectError", "ReadError", "WriteError", "RemoteProtocolError"):
        return True
    return error_type == "ProviderError" and (
        error in ("ReadTimeout", "ConnectTimeout", "WriteTimeout", "PoolTimeout",
                  "ConnectError", "ReadError", "WriteError", "RemoteProtocolError",
                  "empty_output_text", "http_408", "http_409", "http_425", "http_429")
        or error.startswith("http_5"))


def supervise(args):
    campaign = BASE / "revision_openrouter/runs" / args.name
    campaign.mkdir(parents=True, exist_ok=True)
    supervisor_lock = campaign / "supervisor_lock"
    supervisor_lock.mkdir(exist_ok=True)
    with campaign_lock(supervisor_lock):
        write_json(campaign / "supervisor.json", {
            "status": "starting", "pid": os.getpid(), "started_unix": time.time(),
            "workers": args.workers, "max_restarts": args.max_restarts})
        env = {**os.environ, "PYTHONHASHSEED": "0", "PYTHONUTF8": "1"}
        command = [sys.executable, "-X", "utf8", "-u", "-m", "revision_openrouter.run",
                   "--name", args.name, "--limit", str(args.limit),
                   "--workers", str(args.workers)]
        for attempt in range(1, args.max_restarts + 2):
            started = time.time()
            with (campaign / "stdout.log").open("ab") as out, \
                    (campaign / "stderr.log").open("ab") as err:
                child = subprocess.Popen(
                    command, cwd=BASE, env=env, stdout=out, stderr=err,
                    creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0))
                (campaign / "worker_pid.txt").write_text(str(child.pid), encoding="ascii")
                while True:
                    try:
                        code = child.wait(timeout=30)
                        break
                    except subprocess.TimeoutExpired:
                        write_json(campaign / "supervisor.json", {
                            "status": "running", "pid": os.getpid(), "worker_pid": child.pid,
                            "attempt": attempt, "workers": args.workers,
                            "started_unix": started, "updated_unix": time.time()})
            event = {"attempt": attempt, "worker_pid": child.pid,
                     "started_unix": started, "finished_unix": time.time(), "exit_code": code}
            if code == 0 and (campaign / "verification.json").exists():
                verification = read_json(campaign / "verification.json")
                expected = "verified_complete" if args.limit == 1000 else "verified_pilot"
                if verification["status"] == expected:
                    append_jsonl(campaign / "supervisor_attempts.jsonl", {
                        **event, "status": "verified_complete" if args.limit == 1000
                        else "verified_pilot"})
                    write_json(campaign / "supervisor.json", {
                        **event, "status": expected, "pid": os.getpid()})
                    return 0
            failure_path = campaign / "failure.json"
            failure = read_json(failure_path) if failure_path.exists() else {}
            # A failure from an earlier invocation cannot authorize a retry.
            current_failure = failure_path.exists() and failure_path.stat().st_mtime >= started
            retry = current_failure and retryable(failure) and attempt <= args.max_restarts
            append_jsonl(campaign / "supervisor_attempts.jsonl", {
                **event, "status": "retrying" if retry else "needs_attention",
                "error_type": failure.get("error_type"), "error": failure.get("error")})
            if not retry:
                write_json(campaign / "supervisor.json", {
                    **event, "status": "needs_attention", "pid": os.getpid(),
                    "error_type": failure.get("error_type"), "error": failure.get("error")})
                return code or 1
            write_json(campaign / "supervisor.json", {
                **event, "status": "retrying", "pid": os.getpid(),
                "resume_after_unix": time.time() + 30})
            time.sleep(30)
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--max-restarts", type=int, default=12)
    args = parser.parse_args()
    if (not re_safe_name(args.name) or not 1 <= args.limit <= 1000
            or not 1 <= args.workers <= 32 or not 0 <= args.max_restarts <= 24):
        parser.error("Invalid campaign name, limit, worker count or restart limit")
    raise SystemExit(supervise(args))
