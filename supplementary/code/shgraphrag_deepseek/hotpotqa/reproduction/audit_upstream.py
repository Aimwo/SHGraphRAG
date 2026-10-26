"""Record upstream provenance and offline validation without model/database I/O."""

import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "reproduction/audit"
EXPECTED_COMMIT = "3ed1685fefb1295c43c7fca0bb306e4d5fd08e77"


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def literal_settings(text: str) -> dict:
    values = {}
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    values[target.id] = value
    return values


def main() -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    head = git("rev-parse", "HEAD").decode().strip()
    git("cat-file", "-e", EXPECTED_COMMIT + "^{commit}")
    tracked = git("ls-tree", "-r", "--name-only", "-z", EXPECTED_COMMIT).decode().rstrip("\0").split("\0")
    modified = []
    inventory = {}
    syntax_errors = {}
    syntax_count = 0
    for name in tracked:
        original = git("show", EXPECTED_COMMIT + ":" + name)
        current = (ROOT / name).read_bytes()
        if current.replace(b"\r\n", b"\n") != original.replace(b"\r\n", b"\n"):
            modified.append(name)
        inventory[name] = {
            "upstream_sha256": hashlib.sha256(original).hexdigest(),
            "working_sha256": hashlib.sha256(current).hexdigest(),
            "byte_identical": current == original,
            "equal_ignoring_crlf": current.replace(b"\r\n", b"\n") == original.replace(b"\r\n", b"\n"),
            "modified_in_git_diff": name in modified,
        }
        if name.endswith(".py"):
            syntax_count += 1
            try:
                compile(current, name, "exec")
            except SyntaxError as exc:
                syntax_errors[name] = str(exc)
    modified.sort()
    expected_repairs = sorted([
        ".gitignore",
        "config/settings.py", "processor/dataset_processor.py", "build/main.py",
        "build/build_graph.py", "model/get_models.py", "config/neo4jdb.py",
        "graph/structure/struct_builder.py", "graph/extraction/entity_extractor.py",
    ])
    if modified != expected_repairs:
        raise ValueError(f"Unexpected scientific source changes: {modified}")
    original_settings = literal_settings(git("show", EXPECTED_COMMIT + ":config/settings.py").decode("utf-8-sig"))
    current_settings = literal_settings((ROOT / "config/settings.py").read_text(encoding="utf-8-sig"))
    for name, value in original_settings.items():
        if current_settings.get(name) != value:
            raise ValueError(f"Original upstream setting changed: {name}")
    local_import_gaps = []
    main_tree = ast.parse((ROOT / "build/main.py").read_text(encoding="utf-8-sig"))
    for node in main_tree.body:
        if isinstance(node, ast.ImportFrom) and node.module:
            if (ROOT / "build" / (node.module + ".py")).exists() and not (ROOT / (node.module + ".py")).exists():
                local_import_gaps.append(node.module)
    history = git("log", EXPECTED_COMMIT, "--format=%H %s", "--", "qa").decode().strip()
    original_build = ast.parse(git("show", EXPECTED_COMMIT + ":build/build_graph.py").decode("utf-8-sig"))
    active_pruning = sorted({
        node.func.attr
        for node in ast.walk(original_build)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr.startswith("_prune_")
    })
    source_files = [
        path for folder in ("reproduction", "processor")
        for path in (ROOT / folder).glob("*.py")
        if path.relative_to(ROOT).as_posix() not in inventory
    ]
    for path in source_files:
        compile(path.read_bytes(), str(path.relative_to(ROOT)), "exec")
    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "remote": git("remote", "get-url", "origin").decode().strip(),
        "commit": EXPECTED_COMMIT,
        "working_commit": head,
        "branch": git("branch", "--show-current").decode().strip(),
        "shallow": git("rev-parse", "--is-shallow-repository").decode().strip() == "true",
        "commit_count": int(git("rev-list", EXPECTED_COMMIT, "--count").decode()),
        "root_tree": git("ls-tree", "--name-only", EXPECTED_COMMIT).decode().splitlines(),
        "tracked_files": len(tracked),
        "modified_upstream_files": modified,
        "all_original_literal_settings_preserved": True,
        "additional_settings": {
            key: value for key, value in current_settings.items() if key not in original_settings
        },
        "original_python_files_syntax_checked": syntax_count,
        "syntax_errors": syntax_errors,
        "additional_python_files_syntax_checked": len(source_files),
        "upstream_missing_qa": not any(name.startswith("qa/") for name in tracked),
        "upstream_readme_evaluator_exists": "qa/hotpot_evaluate_v1.py" in tracked,
        "qa_commit_history_matches": history.splitlines(),
        "readme_module_entrypoint_local_import_gaps": local_import_gaps,
        "active_pruning_calls": active_pruning,
        "upstream_two_hop_build": "Placeholder #todo after pruning; no implementation added",
        "undeclared_active_dependencies": ["datasets", "langchain_ollama"],
        "dependency_resolution": {
            "upstream_resolved": True,
            "resolved_packages": 182,
            "full_runtime_installed": False,
            "dataset_tools_installed": True,
            "active_benchmark_runtime_installed": True,
            "aiohttp": "3.8.5, built with process-local AIOHTTP_NO_EXTENSIONS=1 for Python 3.12",
            "six": "1.12.0 failed six.moves import on Python 3.12; dataset tools use 1.17.0",
            "full_textract_six_pin_conflict": "Not resolved; no assertion of a fully matching graph environment",
        },
        "sample_validation": json.loads(
            (ROOT / "reproduction/artifacts/musique_seed42_n1000/validation.json").read_text(encoding="utf-8")
        ),
        "processor_validation": json.loads(
            (AUDIT / "processor_validation.json").read_text(encoding="utf-8")
        ),
        "model_api_calls": 0,
        "model_api_tokens": 0,
        "provider_usage_reported": False,
        "credentials_read": False,
        "graph_created_or_cleared": False,
        "full_reproduction_completed": False,
        "source_decision_needed": False,
        "source_combination_approved": "Official build plus supplied local QA/evaluators",
        "python": sys.version,
    }
    write_json(AUDIT / "source_hashes.json", inventory)
    write_json(AUDIT / "UPSTREAM_AUDIT.json", report)
    print(json.dumps({
        "sample_ready": True,
        "modified_upstream_files": modified,
        "source_syntax_errors": syntax_errors,
        "missing_qa_in_tree_and_history": report["upstream_missing_qa"] and not history,
        "model_api_calls": 0,
        "full_reproduction_completed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
