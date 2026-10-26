"""Authenticate embedding log segments against their preserved execution contracts."""
from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path

from revision_openrouter.transport import file_hash, signature


LOG_NAME = "embedding_http_attempts.jsonl"
CLIENT_NAME = "revision_openrouter/embedding_client.py"


class EmbeddingHistoryError(ValueError):
    pass


@dataclass
class EmbeddingHistory:
    contracts: dict
    occurrences: Counter
    segments: list


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _require(passed, message):
    if not passed:
        raise EmbeddingHistoryError(message)


def _prefix(current, original):
    with current.open("rb") as combined, original.open("rb") as preserved:
        while chunk := preserved.read(1024 * 1024):
            if combined.read(len(chunk)) != chunk:
                return False
    return True


def _constant(node, values):
    if isinstance(node, ast.Name):
        return values[node.id]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _constant(node.left, values) + _constant(node.right, values)
    return ast.literal_eval(node)


def _client_constants(stage, provenance):
    entries = [
        (name, digest) for name, digest in provenance["source_sha256"].items()
        if name.replace("\\", "/") == CLIENT_NAME
    ]
    _require(len(entries) == 1, f"{stage}: missing or ambiguous frozen embedding client")
    source = stage / "source_snapshot" / CLIENT_NAME
    _require(source.is_file() and file_hash(source) == entries[0][1],
             f"{stage}: frozen embedding client hash mismatch")
    constants = {}
    # Interpret only simple module constants. Never execute archived source.
    for node in ast.parse(source.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    try:
                        constants[target.id] = _constant(node.value, constants)
                    except (ValueError, TypeError, KeyError):
                        pass
    return constants


def current_contract(config):
    return {
        "identity": {
            "model": config["embedding"],
            "digest": config["embedding_digest"],
            "endpoint": config["embedding_endpoint"],
            "execution_options": config["embedding_execution_options"],
        },
        "options_present": True,
        "max_attempts": 4,
        "overflow_policy": config.get("embedding_overflow_policy"),
    }


def _stage_contract(stage, provenance, reference):
    config = provenance["config"]
    constants = _client_constants(stage, provenance)
    endpoint = constants.get("ENDPOINT")
    options = constants.get("EMBED_OPTIONS")
    _require(endpoint in {
        "http://127.0.0.1:11434/api/embed",
        "http://127.0.0.1:11436/api/embed",
    }, f"{stage}: unsupported frozen embedding endpoint")
    if "embedding_endpoint" in config:
        _require(config["embedding_endpoint"] == endpoint,
                 f"{stage}: endpoint differs between configuration and frozen source")
    _require(options is None or options == {"num_ctx": 8192, "num_batch": 8192},
             f"{stage}: unsupported frozen embedding options")
    if "embedding_execution_options" in config:
        _require(config["embedding_execution_options"] == options,
                 f"{stage}: options differ between configuration and frozen source")
    _require(constants.get("MAX_ATTEMPTS") == 4,
             f"{stage}: unsupported embedding retry bound")
    policy = config.get("embedding_overflow_policy")
    _require(policy == constants.get("OVERFLOW_POLICY"),
             f"{stage}: overflow policy differs between configuration and frozen source")
    _require(all(config.get(key) == reference.get(key) for key in (
        "embedding", "embedding_digest", "embedding_dimension",
        "embedding_normalize", "embedding_truncate",
    )), f"{stage}: embedding model or vector contract changed")
    return {
        "identity": {
            "model": config["embedding"],
            "digest": config["embedding_digest"],
            "endpoint": endpoint,
            "execution_options": options,
        },
        "options_present": "EMBED_OPTIONS" in constants,
        "max_attempts": constants["MAX_ATTEMPTS"],
        "overflow_policy": policy,
    }


def identity_matches(record, contract):
    if contract is None:
        return False
    expected = contract["identity"]
    return (
        all(record.get(key) == value for key, value in expected.items())
        and ("execution_options" in record) == contract["options_present"]
        and type(record.get("attempt")) is int
        and 1 <= record["attempt"] <= contract["max_attempts"]
        and record.get("max_attempts") == contract["max_attempts"]
    )


def request_options_match(request, contract):
    return (contract is not None
            and ("options" in request) == contract["options_present"]
            and request.get("options") == contract["identity"]["execution_options"])


def load_authenticated_embedding_history(campaign, config):
    """Bind each exact logged row to its original stage; reject broken lineage."""
    campaign = Path(campaign)
    stages = []
    stage = campaign
    for depth in range(32):
        provenance_path = stage / "provenance.json"
        _require(provenance_path.is_file(), f"{stage}: missing provenance")
        provenance = _read(provenance_path)
        _require(signature(provenance["config"]) == provenance.get("config_sha256"),
                 f"{stage}: configuration signature mismatch")
        if depth == 0:
            _require(provenance["config"] == config,
                     "Current audit configuration differs from frozen provenance")
        manifest_path = stage / "recovery_manifest.json"
        manifest = _read(manifest_path) if manifest_path.is_file() else None
        parent = stage / ("recovery_origin" if depth == 0 else "earlier_transport_origin")
        log = stage / LOG_NAME
        inherited_bytes = 0
        if manifest is not None:
            _require(parent.is_dir() and (parent / "provenance.json").is_file(),
                     f"{stage}: missing preserved parent")
            _require(file_hash(parent / "provenance.json")
                     == manifest.get("origin_provenance_sha256"),
                     f"{stage}: parent provenance hash mismatch")
            parent_manifest = parent / "recovery_manifest.json"
            expected_manifest = manifest.get("parent_recovery_manifest_sha256")
            _require((parent_manifest.is_file()
                      and file_hash(parent_manifest) == expected_manifest)
                     or (not parent_manifest.exists() and expected_manifest is None),
                     f"{stage}: parent recovery manifest hash mismatch")
            inherited = manifest.get("inherited_history_sha256", {})
            expected_log = inherited.get(LOG_NAME)
            parent_log = parent / LOG_NAME
            if parent_log.is_file():
                _require(expected_log is not None and file_hash(parent_log) == expected_log,
                         f"{stage}: archived embedding log hash mismatch")
                _require(log.is_file() and _prefix(log, parent_log),
                         f"{stage}: embedding history prefix changed")
                inherited_bytes = parent_log.stat().st_size
                if inherited_bytes:
                    with parent_log.open("rb") as stream:
                        stream.seek(-1, 2)
                        _require(stream.read(1) == b"\n",
                                 f"{stage}: incomplete inherited embedding log row")
            else:
                _require(expected_log is None,
                         f"{stage}: declared inherited embedding log missing")
        else:
            _require(not parent.exists(), f"{stage}: unlinked preserved parent")

        contract = _stage_contract(stage, provenance, config) if log.is_file() else None
        stages.append((stage, log, inherited_bytes, contract, provenance, manifest))
        if manifest is None:
            break
        stage = parent
    else:
        raise EmbeddingHistoryError("Embedding recovery lineage exceeds 32 stages")

    contracts, occurrences, segments = {}, Counter(), []
    for stage, log, inherited_bytes, contract, provenance, manifest in reversed(stages):
        added = 0
        if log.is_file():
            with log.open("rb") as stream:
                stream.seek(inherited_bytes)
                for line in stream:
                    if not line.strip():
                        continue
                    _require(line.endswith(b"\n"), f"{stage}: incomplete embedding log row")
                    try:
                        row = json.loads(line)
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise EmbeddingHistoryError(f"{stage}: malformed embedding log") from exc
                    key = signature(row)
                    _require(key not in contracts,
                             f"{stage}: duplicate exact row in recovered embedding history")
                    contracts[key] = contract
                    occurrences[key] += 1
                    added += 1
        segments.append({
            "directory": str(stage.relative_to(campaign)),
            "scope": (manifest or {}).get("scope", "original"),
            "config_sha256": provenance["config_sha256"],
            "provenance_sha256": file_hash(stage / "provenance.json"),
            "log_sha256": file_hash(log) if log.is_file() else None,
            "inherited_log_bytes": inherited_bytes,
            "new_attempts": added,
            "contract": contract,
        })
    return EmbeddingHistory(contracts, occurrences, segments)
