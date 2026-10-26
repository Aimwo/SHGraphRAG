"""Pure offline candidate for an explicitly approved capacity recovery.

This module neither migrates files nor launches a run. It prepares one cache
entry only after proving its exact original text request and terminal response.
Physical request identities retain their original scope. A future recovery
must preserve and audit the referenced original history separately, count its
physical attempts once, and rebuild every embedding vector.
"""
from __future__ import annotations

import copy
import hashlib
import json


TERMINAL = {
    "completed", "provider_filtered", "output_truncated", "output_incomplete",
}
IDENTITY_KEYS = (
    "provider", "base_url", "model", "credential_id",
    "generation_parameters", "allowed_returned_models", "embedding",
)


def signature(value):
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def scopes(source_config, target_config):
    """Permit precisely the already proposed independent capacity change."""
    require(source_config["embedding"]["options"] == {},
            "Unexpected original embedding options")
    require(target_config["embedding"]["options"] == {"num_batch": 8192},
            "Unexpected proposed embedding options")
    restored = copy.deepcopy(target_config)
    restored["embedding"]["options"] = {}
    require(source_config["max_in_flight"] == 32
            and target_config["max_in_flight"] == 100,
            "Unexpected concurrency transition")
    restored["max_in_flight"] = source_config["max_in_flight"]
    restored["authorization"] = source_config["authorization"]
    require(restored == source_config,
            "Recovery changes more than the authorized capacity/scheduling settings")
    return tuple(signature({k: config[k] for k in IDENTITY_KEYS})
                 for config in (source_config, target_config))


def prepare_entry(*, source_cache_bytes, source_artifact_bytes, mapping,
                  source_config, target_config):
    """Return candidate cache data; leave all physical evidence unchanged."""
    old_scope, new_scope = scopes(source_config, target_config)
    require(hashlib.sha256(source_cache_bytes).hexdigest()
            == mapping["source_cache_sha256"], "Original cache bytes changed")
    saved = json.loads(source_cache_bytes)
    artifact = json.loads(source_artifact_bytes)
    row = artifact["record"]
    body = artifact["request"]
    endpoint = source_config["base_url"].rstrip("/") + "/chat/completions"
    payload = {k: v for k, v in artifact.items() if k != "record"}
    require(signature(payload) == row["raw_payload_sha256"]
            == mapping["source_raw_payload_sha256"],
            "Original physical input/response hash changed")
    require(row["kind"] == "text" and row["http_status"] == 200
            and row["endpoint"] == artifact["endpoint"] == endpoint,
            "Original text endpoint or HTTP outcome differs")
    require(body["model"] == row["requested_model"] == source_config["model"]
            and row["credential_id"] == source_config["credential_id"],
            "Original text model or credential identity differs")
    original_request = signature({
        "scope": old_scope, "endpoint": endpoint, "body": body,
    })
    old_key = signature({"scope": old_scope, "kind": "text", "body": body})
    new_key = signature({"scope": new_scope, "kind": "text", "body": body})
    require(row["scope_sha256"] == saved["scope_sha256"]
            == mapping["source_scope_sha256"] == old_scope,
            "Original physical scope differs")
    require(row["request_sha256"] == saved["request_sha256"]
            == mapping["source_request_sha256"] == original_request,
            "Original physical request identity differs")
    require(saved["cache_key"] == old_key
            and mapping["source_cache"] == "cache/responses/" + old_key + ".json",
            "Original cache key or path differs")
    require(row["attempt_id"] == mapping["source_attempt_id"]
            and row["raw_artifact"] == mapping["source_artifact"]
            and saved.get("raw_artifact", row["raw_artifact"]) == row["raw_artifact"],
            "Original attempt reference differs")
    require(mapping["request_body_sha256"] == signature(body)
            and mapping["proposed_scope_sha256"] == new_scope
            and mapping["proposed_cache_key"] == new_key,
            "Proposed mapping does not match the unchanged full text request")
    response = artifact["response_json"]
    require(response.get("model") == row.get("returned_model")
            and response.get("model") in source_config["allowed_returned_models"]
            and response.get("id") == saved.get("response_id")
            == mapping.get("response_id"),
            "Original returned model or response identity differs")
    require(len(response["choices"]) == 1, "Unexpected completion count")
    choice = response["choices"][0]
    message = choice.get("message") or {}
    content = message.get("content") or ""
    require(isinstance(content, str), "Original visible response is not text")
    finish = choice.get("finish_reason")
    if message.get("refusal") or finish == "content_filter":
        status = "provider_filtered"
    elif finish == "length":
        status = "output_truncated"
    elif finish != "stop":
        status = "output_incomplete"
    else:
        require(bool(content.strip()), "Empty stopped response is not reusable")
        status = "completed"
    require(status in TERMINAL
            and status == row["status"] == saved["status"] == mapping["status"],
            "Original terminal outcome differs")
    require(saved["text" if status == "completed" else "raw_text"] == content
            and saved["content_sha256"] == mapping["content_sha256"]
            == signature(content), "Original visible text or content hash differs")
    require(mapping["stage"] == row["stage"], "Original request stage differs")
    target = copy.deepcopy(saved)
    target["cache_key"] = new_key
    target["scope_sha256"] = new_scope
    target["embedding_capacity_recovery"] = {
        key: mapping[key] for key in (
            "source_cache", "source_cache_sha256", "source_attempt_id",
            "source_request_sha256", "source_scope_sha256", "source_artifact",
            "source_raw_payload_sha256", "request_body_sha256",
        )
    }
    # request_sha256 deliberately remains the ORIGINAL physical request hash.
    # No new physical attempt, response, usage or artifact is manufactured.
    return target


def audit_entry(target, **original):
    """Require exact rekeying, including unavailable outcomes and origin IDs."""
    require(target == prepare_entry(**original),
            "Imported candidate cache differs from its proven original")
    return {
        "status": target["status"],
        "cache_key": target["cache_key"],
        "original_physical_request_sha256": target["request_sha256"],
        "new_physical_attempts": 0,
        "imported_vectors": 0,
    }
