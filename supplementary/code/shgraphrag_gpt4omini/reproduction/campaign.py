"""Sequential dataset campaign with one bounded OpenRouter request pool."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .client import EFFORT, MODEL, PROVIDER, RUNTIME_DIRECTORY, write_json
from .data import DATASETS, sha256
from .report import summarize
from .verify_run import verify


ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def campaign_lock():
    location = ROOT / RUNTIME_DIRECTORY / "campaign.lock"
    location.parent.mkdir(parents=True, exist_ok=True)
    with location.open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--wait-for-run", type=Path)
    args = parser.parse_args()
    base = ROOT / "reproduction/runs" / args.tag
    base.mkdir(parents=True, exist_ok=True)
    state = {
        "status": "initializing", "pid": os.getpid(), "limit": args.limit,
        "workers": args.workers, "datasets": args.datasets, "completed": [],
        "model": MODEL, "provider": PROVIDER, "reasoning_effort": EFFORT,
        "started_unix": time.time(),
    }
    def save(**fields):
        state.update(fields, updated_unix=time.time())
        write_json(base / "campaign.json", state)
    with campaign_lock():
        if args.wait_for_run:
            save(status="waiting_for_existing_pilot", waiting_for=str(args.wait_for_run))
            while not (args.wait_for_run / "evaluation/metrics.json").exists():
                status = args.wait_for_run / "status.json"
                if status.exists():
                    current = json.loads(status.read_text(encoding="utf-8"))
                    if current.get("status") == "failed":
                        save(status="needs_attention", error="existing_pilot_failed")
                        return 1
                time.sleep(2)
        initial_code = {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")}
        write_json(base / "campaign_code_lock.json", initial_code)
        environment = os.environ.copy()
        environment.update(PYTHONUTF8="1", OPENBLAS_NUM_THREADS="1")
        executable = Path(sys.executable).with_name("python.exe") if os.name == "nt" else Path(sys.executable)
        for dataset in args.datasets:
            current_code = {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")}
            if current_code != initial_code:
                save(status="needs_attention", error="code_changed_during_campaign")
                return 1
            destination = base / dataset
            save(status="running", current_dataset=dataset, run=str(destination))
            command = [str(executable), "-u", "-m", "reproduction.runner",
                       "--dataset", dataset, "--limit", str(args.limit),
                       "--workers", str(args.workers), "--output", str(destination)]
            with (base / f"{dataset}.log").open("ab") as log:
                result = subprocess.run(command, cwd=ROOT, env=environment,
                                        stdout=log, stderr=subprocess.STDOUT,
                                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            summary = summarize(destination)
            if result.returncode or not summary["complete"]:
                save(status="needs_attention", exit_code=result.returncode,
                     error="dataset_pipeline_incomplete")
                return 1
            verify(destination)
            state["completed"].append(dataset)
            save(status="dataset_completed")
        save(status="completed", completed_unix=time.time())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
