"""Atomic, input-bound checkpoints for completed independent document merges."""

import hashlib
import json
import os
from pathlib import Path

from reproduction.runtime import ROOT, append_jsonl, run_root, utc_now, write_json


class MergeCheckpoint:
    def __init__(self, document, threshold):
        from reproduction.relay_config import load_relay_config, config_identity
        relay_config = load_relay_config(require_ready=False)
        self.document = document
        root = run_root()
        preflight = root / "local_preflight.json"
        embedding_model = (
            json.loads(preflight.read_text(encoding="utf-8")).get("local_embedding_model")
            if preflight.exists() else {"name": os.getenv("OLLAMA_EMBEDDINGS_MODEL")}
        )
        signature = {
            "question_id": document["question_id"],
            "chunks": (
                list(document["chunks"]) if document.get("chunks") is not None else None
            ),
            "entity_data": document.get("entity_data"),
            "unmerged_structured_data": document.get("structured_data"),
            "threshold": threshold, "embedding_model": embedding_model,
            "relay_config_sha256": config_identity(relay_config),
            "credential_id": relay_config["credential_id"],
            "text_model": relay_config["model"],
            "build_source_sha256": hashlib.sha256(
                (ROOT / "build/build_graph.py").read_bytes()
            ).hexdigest(),
        }
        self.key = hashlib.sha256(
            json.dumps(signature, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        self.path = root / "cache/merged_documents" / (self.key + ".json")

    def restore(self):
        if not self.path.exists():
            return False
        saved = json.loads(self.path.read_text(encoding="utf-8"))
        if saved["input_signature_sha256"] != self.key:
            raise ValueError("Document merge checkpoint has a different input signature")
        encoded = json.dumps(saved["structured_data"], sort_keys=True, ensure_ascii=False)
        if hashlib.sha256(encoded.encode("utf-8")).hexdigest() != saved["result_sha256"]:
            raise ValueError("Document merge checkpoint content hash mismatch")
        self.document["structured_data"] = saved["structured_data"]
        self._event("reused")
        return True

    def save(self):
        result = self.document["structured_data"]
        encoded = json.dumps(result, sort_keys=True, ensure_ascii=False)
        write_json(self.path, {
            "input_signature_sha256": self.key,
            "result_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
            "structured_data": result, "completed_at": utc_now(),
        })
        self._event("completed")

    def _event(self, status):
        append_jsonl(run_root() / "document_merge_events.jsonl", {
            "at": utc_now(), "question_id": self.document["question_id"],
            "status": status, "input_signature_sha256": self.key,
        })
