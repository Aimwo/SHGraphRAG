"""Run reproducible offline checks and save their complete output."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from unittest.mock import patch

from .common import ROOT, file_hash, write_json


def main():
    audit = ROOT / "audit"
    audit.mkdir(exist_ok=True)
    environment = {**os.environ, "PYTHONUTF8": "1", "PYTHONHASHSEED": "42"}
    started = time.time()
    command = [sys.executable, "-X", "utf8", "-m", "unittest",
               "experiment.test_experiment", "experiment.test_additional_datasets", "experiment.test_runtime_recovery", "-v"]
    result = subprocess.run(
        command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
        timeout=180, creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0))
    log = audit / "offline_tests.log"
    log.write_text(result.stdout, encoding="utf-8")
    syntax = {}
    for source in sorted([
            *(ROOT / "experiment").glob("*.py"),
            *(ROOT / "hypergraphrag").rglob("*.py")]):
        relative = source.relative_to(ROOT).as_posix()
        try:
            compile(source.read_text(encoding="utf-8"), str(source), "exec")
            syntax[relative] = {"passed": True}
        except SyntaxError as exc:
            syntax[relative] = {"passed": False, "line": exc.lineno, "message": exc.msg}
    selected = {k: v for k, v in syntax.items() if "/kg/" not in k}
    optional_issues = {k: v for k, v in syntax.items() if "/kg/" in k and not v["passed"]}
    syntax_ok = all(v["passed"] for v in selected.values())
    write_json(audit / "syntax.json", {
        "selected_execution_path_passed": syntax_ok, "files": syntax,
        "unused_optional_adapter_issues": optional_issues,
        "scope": "Selected NetworkX/JsonKV/NanoVectorDB path; kg adapters are lazy and unused.",
        "note": "The preserved upstream TiDB adapter uses Python 3.12 f-string syntax; "
                "it is not imported by this Python 3.11 experiment."})
    from .run import check
    with patch("experiment.run.read_authorized_key", side_effect=AssertionError(
            "Offline checks may not load the saved credential")), patch(
            "httpx.AsyncHTTPTransport.handle_async_request", side_effect=AssertionError(
                "Offline checks may not access the network")):
        readiness = check()
    report = {
        "status": "passed" if result.returncode == 0 and syntax_ok else "failed",
        "command": command, "test_returncode": result.returncode,
        "syntax_ok": syntax_ok, "started_unix": started, "finished_unix": time.time(),
        "unused_optional_adapter_issues": optional_issues,
        "test_output": log.relative_to(ROOT).as_posix(), "test_output_sha256": file_hash(log),
        "readiness_status": readiness["status"], "run_blockers": readiness["blockers"],
        "live_model_requests_by_checks": 0, "saved_credential_loaded_by_checks": False,
        "note": "HTTP tests use MockTransport and a dummy key; default HTTP transport is forbidden.",
    }
    write_json(audit / "offline_checks.json", report)
    print(result.stdout[-10000:])
    print(json.dumps(report, ensure_ascii=True, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
