import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from reproduction.merge_checkpoint import MergeCheckpoint
from reproduction.runtime import ROOT
from reproduction.relay_config import load_relay_config


class MergeCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(
            prefix="offline_merge_", dir=ROOT / "reproduction/runs"
        )
        self.environment = patch.dict(os.environ, {
            "SHGRAPH_RUN_ROOT": self.temporary.name,
            "OLLAMA_EMBEDDINGS_MODEL": "bge-m3:latest",
        })
        self.environment.start()

    def tearDown(self):
        self.environment.stop()
        self.temporary.cleanup()

    def test_completed_document_restores_and_changed_input_is_rejected(self):
        original = {
            "question_id": "frozen-a", "chunks": ["original passage"],
            "entity_data": ["original tuple response"],
            "structured_data": [{"nodes": {"a": {"description": "first"}}, "relationships": []}],
        }
        doc = copy.deepcopy(original)
        checkpoint = MergeCheckpoint(doc, 0.95)
        self.assertFalse(checkpoint.restore())
        doc["structured_data"][0]["nodes"]["a"]["description"] = "merged result"
        checkpoint.save()
        resumed = copy.deepcopy(original)
        self.assertTrue(MergeCheckpoint(resumed, 0.95).restore())
        self.assertEqual(resumed["structured_data"], doc["structured_data"])
        changed = copy.deepcopy(original)
        changed["chunks"] = ["different passage"]
        self.assertFalse(MergeCheckpoint(changed, 0.95).restore())
        self.assertFalse(MergeCheckpoint(copy.deepcopy(original), 0.94).restore())

    def test_parquet_array_chunks_restore_without_mutating_the_live_document(self):
        original = {
            "question_id": "parquet-array",
            "chunks": np.array(["原始段落", "second passage"], dtype=object),
            "entity_data": ["original tuple response"],
            "structured_data": [
                {"nodes": {"a": {"description": "first"}}, "relationships": []}
            ],
        }
        doc = copy.deepcopy(original)
        live_chunks = doc["chunks"]
        try:
            checkpoint = MergeCheckpoint(doc, 0.95)
        except TypeError as error:
            self.fail(f"Parquet array chunks must support checkpoints: {error}")
        doc["structured_data"][0]["nodes"]["a"]["description"] = "merged result"
        checkpoint.save()

        resumed = copy.deepcopy(original)
        resumed["chunks"] = ["原始段落", "second passage"]
        self.assertTrue(MergeCheckpoint(resumed, 0.95).restore())
        self.assertEqual(
            resumed["structured_data"],
            [{"nodes": {"a": {"description": "merged result"}}, "relationships": []}],
        )
        self.assertIs(doc["chunks"], live_chunks)
        np.testing.assert_array_equal(live_chunks, ["原始段落", "second passage"])

    def test_a_middle_array_value_change_invalidates_the_checkpoint(self):
        original = {
            "question_id": "long-parquet-array",
            "chunks": np.array([f"passage {i}" for i in range(1500)], dtype=object),
            "entity_data": [],
            "structured_data": [],
        }
        try:
            checkpoint = MergeCheckpoint(original, 0.95)
        except TypeError as error:
            self.fail(f"Parquet array chunks must support checkpoints: {error}")
        checkpoint.save()
        changed = copy.deepcopy(original)
        changed["chunks"][750] = "different middle passage"
        self.assertFalse(MergeCheckpoint(changed, 0.95).restore())

    def test_unsupported_chunk_values_are_not_stringified(self):
        doc = {
            "question_id": "invalid-object",
            "chunks": [object()],
            "entity_data": [],
            "structured_data": [],
        }
        with self.assertRaises(TypeError):
            MergeCheckpoint(doc, 0.95)

    def test_capacity_only_change_restores_exact_legacy_checkpoint_and_records_origin(self):
        current = load_relay_config(require_ready=False)
        old = {**current, "max_in_flight": 64}
        original = {
            "question_id": "capacity-change", "chunks": ["same candidate"],
            "entity_data": ["same tuples"], "structured_data": [{"nodes": {}}],
        }
        doc = copy.deepcopy(original)
        with patch("reproduction.relay_config.load_relay_config", return_value=old):
            checkpoint = MergeCheckpoint(doc, 0.95)
            doc["structured_data"] = [{"nodes": {"entity": {"description": "merged"}}}]
            checkpoint.save()
        resumed = copy.deepcopy(original)
        new_checkpoint = MergeCheckpoint(resumed, 0.95)
        self.assertNotEqual(checkpoint.key, new_checkpoint.key)
        self.assertTrue(new_checkpoint.restore())
        self.assertEqual(resumed["structured_data"], doc["structured_data"])
        self.assertFalse(new_checkpoint.path.exists())
        events = [
            json.loads(line) for line in (
                Path(self.temporary.name) / "document_merge_events.jsonl"
            ).read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(events[-1]["input_signature_sha256"], checkpoint.key)
        self.assertEqual(events[-1]["current_input_signature_sha256"], new_checkpoint.key)
        self.assertEqual(Path(events[-1]["checkpoint_path"]), checkpoint.path)
        self.assertEqual(
            events[-1]["scheduling_compatibility_reason"],
            "user-approved-request-capacity64-to100-20261025",
        )
        for field, value in [
            ("model", "changed-model"),
            ("credential_id", "changed-credential"),
            ("generation_parameters", {"temperature": 0.2}),
        ]:
            with self.subTest(field=field):
                changed_config = {**current, field: value}
                with patch(
                    "reproduction.relay_config.load_relay_config",
                    return_value=changed_config,
                ):
                    self.assertFalse(MergeCheckpoint(copy.deepcopy(original), 0.95).restore())
        changed = copy.deepcopy(original)
        changed["chunks"] = ["different candidate"]
        self.assertFalse(MergeCheckpoint(changed, 0.95).restore())
        self.assertFalse(MergeCheckpoint(copy.deepcopy(original), 0.94).restore())

    def test_legacy_compatibility_keeps_result_integrity_verification(self):
        current = load_relay_config(require_ready=False)
        doc = {
            "question_id": "corrupted-legacy", "chunks": ["same candidate"],
            "entity_data": [], "structured_data": [],
        }
        with patch(
            "reproduction.relay_config.load_relay_config",
            return_value={**current, "max_in_flight": 64},
        ):
            checkpoint = MergeCheckpoint(copy.deepcopy(doc), 0.95)
            checkpoint.save()
        saved = json.loads(checkpoint.path.read_text(encoding="utf-8"))
        saved["structured_data"] = [{"tampered": True}]
        checkpoint.path.write_text(json.dumps(saved), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "content hash mismatch"):
            MergeCheckpoint(copy.deepcopy(doc), 0.95).restore()


if __name__ == "__main__":
    unittest.main()
