"""Passive experiment telemetry, excluding environment values and credentials."""
from __future__ import annotations

import asyncio
import ctypes
from ctypes import wintypes
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time

from .transport import append_jsonl, write_json


def process_resources():
    sample = {"time_unix": time.time(), "pid": os.getpid(),
              "process_cpu_s": time.process_time()}
    if os.name != "nt":
        return sample

    class MemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
            *[(name, ctypes.c_size_t) for name in (
                "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage",
                "PrivateUsage")],
        ]

    class IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    handle = kernel.GetCurrentProcess()
    memory = MemoryCounters()
    memory.cb = ctypes.sizeof(memory)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(MemoryCounters), wintypes.DWORD]
    if psapi.GetProcessMemoryInfo(handle, ctypes.byref(memory), memory.cb):
        sample.update(
            resident_bytes=memory.WorkingSetSize,
            peak_resident_bytes=memory.PeakWorkingSetSize,
            private_bytes=memory.PrivateUsage,
            peak_commit_bytes=memory.PeakPagefileUsage)
    counters = IoCounters()
    kernel.GetProcessIoCounters.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(IoCounters)]
    if kernel.GetProcessIoCounters(handle, ctypes.byref(counters)):
        sample.update(
            read_bytes=counters.ReadTransferCount,
            write_bytes=counters.WriteTransferCount)
    return sample


def gpu_snapshot():
    if not shutil.which("nvidia-smi"):
        return {"available": False}
    try:
        result = subprocess.run([
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu,utilization.memory,power.draw",
            "--format=csv,noheader,nounits",
        ], capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=10, creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0))
        return {
            "available": result.returncode == 0,
            "columns": ["name", "driver_version", "memory_total_MiB", "memory_used_MiB",
                        "gpu_utilization_percent", "memory_utilization_percent", "power_W"],
            "rows": result.stdout.strip().splitlines(),
            "note": "Whole shared GPU, including the external embedding service.",
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error_type": type(exc).__name__}


def software_snapshot():
    result = {
        "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "processor": platform.processor(),
        "cpu_logical_count": os.cpu_count(),
        "packages": {distribution.metadata["Name"]: distribution.version
                     for distribution in importlib.metadata.distributions()
                     if distribution.metadata.get("Name")},
        "timing_clock": "time.perf_counter monotonic wall time; time.time UTC timestamps",
        "resource_scope": "Worker-process CPU/RAM/IO; GPU values cover all shared GPU clients.",
    }
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD)] + [
                (name, ctypes.c_ulonglong) for name in (
                    "ullTotalPhys", "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile",
                    "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual")]
        memory = MemoryStatus()
        memory.dwLength = ctypes.sizeof(memory)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
            result["system_memory_bytes"] = memory.ullTotalPhys
    return result


async def resource_monitor(campaign, state, transport):
    count = 0
    while True:
        sample = {
            **process_resources(), "dataset": state.get("dataset"),
            "phase": state.get("phase"), "llm_calls": transport.calls,
            "cache_hits": transport.hits,
            "llm_active": transport.active_calls,
            "llm_peak_active": transport.peak_active_calls,
        }
        if count % 3 == 0:
            sample["gpu"] = await asyncio.to_thread(gpu_snapshot)
        append_jsonl(campaign / "resource_samples.jsonl", sample)
        count += 1
        await asyncio.sleep(10)


def mark_phase(campaign, dataset, phase, **details):
    append_jsonl(campaign / "phase_events.jsonl", {
        "time_unix": time.time(), "dataset": dataset, "phase": phase, **details})


def save_environment(campaign):
    observed = software_snapshot()
    path = campaign / "environment.json"
    if path.exists():
        original = json.loads(path.read_text(encoding="utf-8"))
        if original["packages"] != observed["packages"] or original["python"] != observed["python"]:
            raise RuntimeError("Interpreter/dependency versions changed: use a new run")
    else:
        write_json(path, observed)
