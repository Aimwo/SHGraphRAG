"""Validate published release bytes and documentation without dependencies/APIs."""
from __future__ import annotations

import argparse
import ast
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote

SUPPLEMENT = Path(__file__).resolve().parents[1]
REPOSITORY = SUPPLEMENT.parent
EXCLUDED_PARTS = {
    ".git", "__pycache__", ".runtime", ".cache", "node_modules", "tokenizer_cache",
    "inputs", "runs", "run", "artifacts", "dataset", "files", "cache", "audit", "launches",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DocumentationHTMLLinks(HTMLParser):
    """Collect navigation and badge paths in GitHub README HTML."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.targets = []

    def handle_starttag(self, tag, attrs):
        attribute = {"a": "href", "img": "src"}.get(tag)
        if attribute:
            self.targets.extend(value for key, value in attrs
                                if key == attribute and value)


def check_links(paths):
    errors = []
    for path in paths:
        if path.name == "UPSTREAM_README.md":
            # Verbatim historical README retains its original image/navigation.
            continue
        text = path.read_text(encoding="utf-8-sig")
        html_links = DocumentationHTMLLinks()
        html_links.feed(text)
        targets = re.findall(r"!?\[[^\]\n]*\]\(([^)\n]+)\)", text)
        for target in [*targets, *html_links.targets]:
            target = target.split(" ", 1)[0].strip("<>")
            if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            target = unquote(target.split("#", 1)[0])
            resolved = (path.parent / target).resolve()
            if not resolved.is_relative_to(REPOSITORY) or not resolved.exists():
                errors.append({"file": path.relative_to(REPOSITORY).as_posix(), "target": target})
    return errors


def check_record_data(value, prefix=()):
    """Historical receipts may contain counts/hashes, never benchmark rows."""
    errors = []
    if isinstance(value, dict):
        for key, item in value.items():
            current = (*prefix, key)
            if (key in {"per_question", "per_sample", "call_ids",
                        "failure_attempts", "embedding_native_usage"}
                    or key in {"question", "answer", "prediction", "response_text",
                               "raw_response"} and isinstance(item, str)):
                errors.append("/".join(current))
            errors.extend(check_record_data(item, current))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            errors.extend(check_record_data(item, (*prefix, str(index))))
    return errors


def validate():
    manifest = json.loads((SUPPLEMENT / "provenance/source_manifest.json").read_text(encoding="utf-8"))
    errors, syntax_notes, json_count, checked = [], [], 0, 0
    for relative, entry in manifest["files"].items():
        path = SUPPLEMENT / relative
        if not path.is_file() or sha256(path) != entry["published_sha256"]:
            errors.append({"check": "file_hash", "file": relative})
            continue
        checked += 1
        if path.suffix == ".json":
            try:
                value = json.loads(path.read_text(encoding="utf-8-sig"))
                json_count += 1
                if relative.startswith("records/"):
                    errors.extend({"check": "per_example_or_attempt_data",
                                   "file": relative, "field": field}
                                  for field in check_record_data(value))
            except ValueError:
                errors.append({"check": "json", "file": relative})
        if path.suffix == ".py":
            try:
                ast.parse(path.read_text(encoding="utf-8-sig"))
            except SyntaxError as exc:
                if "/hypergraphrag/kg/" in relative:
                    syntax_notes.append({"file": relative, "line": exc.lineno,
                                         "note": "Unused optional adapter may require Python 3.12"})
                else:
                    errors.append({"check": "python_syntax", "file": relative, "line": exc.lineno})
        if (relative.startswith("code/") and any(part in EXCLUDED_PARTS for part in Path(relative).parts)
                or path.suffix in {".parquet", ".npy", ".npz", ".sqlite", ".sqlite3", ".zip"}
                or path.name == ".env" or path.name.endswith(".dpapi.json")):
            errors.append({"check": "excluded_runtime_or_data_file", "file": relative})
    for relative, entry in manifest.get("root_files", {}).items():
        path = REPOSITORY / relative
        if not path.is_file() or sha256(path) != entry["published_sha256"]:
            errors.append({"check": "root_file_hash", "file": relative})
    for seed in (42, 20261020):
        for dataset in ("hotpotqa", "2wiki", "musique"):
            path = SUPPLEMENT / "protocols" / f"seed{seed}" / dataset / "ids.json"
            ids = json.loads(path.read_text(encoding="utf-8"))
            if len(ids) != 1000 or len(set(ids)) != 1000 or not all(isinstance(x, str) for x in ids):
                errors.append({"check": "protocol_ids", "seed": seed, "dataset": dataset})
    docs = [REPOSITORY / "README.md", REPOSITORY / "README.zh-CN.md",
            *[SUPPLEMENT / relative for relative in manifest["files"] if relative.endswith(".md")]]
    links = check_links(docs)
    errors.extend({"check": "relative_link", **item} for item in links)
    return {
        "status": "passed" if not errors else "failed",
        "published_files_checked": checked, "json_files_parsed": json_count,
        "protocol_lists_checked": 6, "markdown_files_checked": len(docs),
        "optional_adapter_syntax_notes": syntax_notes, "errors": errors,
        "model_calls": 0, "network_requests": 0,
        "scope": "Release bytes and static documentation only; historical inference not repeated.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate()
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["status"] == "passed" else 1)
