"""Measured runtime and hardware provenance without inspecting credentials."""
from __future__ import annotations

from datetime import datetime
import importlib.metadata
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import threading
import time

import psutil

from .client import append_jsonl, write_json


class RunTelemetry:
    def __init__(self, root: Path, interval: float = 5):
        self.root = root
        self.interval = interval
        self.stop = threading.Event()
        self.process = psutil.Process()
        self.thread = None
        self.peaks = {"process_rss_bytes": 0, "process_cpu_percent": 0,
                      "host_memory_used_bytes": 0}
        self.samples = 0

    def sample(self):
        memory = self.process.memory_info()
        cpu = self.process.cpu_times()
        io = self.process.io_counters()
        host_memory = psutil.virtual_memory()
        record = {
            "time_unix": time.time(), "pid": self.process.pid,
            "process_rss_bytes": memory.rss, "process_vms_bytes": memory.vms,
            "process_cpu_percent": self.process.cpu_percent(),
            "process_cpu_user_s": cpu.user, "process_cpu_system_s": cpu.system,
            "process_threads": self.process.num_threads(),
            "process_io_read_bytes": io.read_bytes,
            "process_io_write_bytes": io.write_bytes,
            "host_cpu_percent": psutil.cpu_percent(),
            "host_memory_used_bytes": host_memory.used,
            "host_memory_available_bytes": host_memory.available,
            "worktree_volume_free_bytes": shutil.disk_usage(self.root).free,
        }
        for key in self.peaks:
            self.peaks[key] = max(self.peaks[key], record[key])
        self.samples += 1
        append_jsonl(self.root / "resources.jsonl", record)

    def watch(self):
        while not self.stop.wait(self.interval):
            self.sample()

    def __enter__(self):
        self.started = time.perf_counter()
        self.root.mkdir(parents=True, exist_ok=True)
        root = Path(__file__).resolve().parents[1]
        metadata = {
            "recorded_at": datetime.now().astimezone().isoformat(),
            "python": sys.version, "python_executable": sys.executable,
            "python_prefix": sys.prefix, "os": platform.platform(),
            "machine": platform.machine(), "hostname": platform.node(),
            "cpu_model": platform.processor(),
            "cpu_physical_cores": psutil.cpu_count(logical=False),
            "cpu_logical_cores": psutil.cpu_count(),
            "ram_total_bytes": psutil.virtual_memory().total,
            "worktree_volume_free_bytes": shutil.disk_usage(self.root).free,
            "package_versions": dict(sorted(
                (d.metadata["Name"], d.version)
                for d in importlib.metadata.distributions())),
            "numeric_thread_settings": {
                k: os.environ.get(k) for k in
                ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "resource_sample_interval_s": self.interval,
            "resource_scope": (
                "Process measurements describe this dataset runner. Shared Ollama "
                "and other applications are not attributed to it. Host measurements "
                "include unrelated workloads; GPU utilization is not sampled."),
            "llm_determinism": (
                "temperature=0; no provider deterministic guarantee or seeded "
                "bitwise replay is claimed. Exact requests and all accepted outputs "
                "are retained for artifact-level reproduction."),
        }
        write_json(self.root / "runtime_environment.json", metadata)
        self.sample()
        self.thread = threading.Thread(target=self.watch, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, error_type, error, traceback):
        self.stop.set()
        self.thread.join(timeout=self.interval + 1)
        self.sample()
        summary = {
            "elapsed_s": time.perf_counter() - self.started,
            "status": "completed" if error_type is None else "failed",
            "error_type": error_type.__name__ if error_type else None,
            "sample_interval_s": self.interval, "samples": self.samples,
            "sampled_peaks": self.peaks,
            "sampling_caveat": "Peaks between samples may be missed.",
        }
        append_jsonl(self.root / "execution_sessions.jsonl", summary)
        write_json(self.root / "resource_summary.json", summary)
