"""Catch reordered metadata and wrong vector inputs before formal acceptance."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from reproduction.client import signature, write_json
from reproduction.helpers.audit_index_provenance import (
    DIGEST, audit_one, input_fingerprint)


class IndexProvenanceTests(unittest.TestCase):
    def test_streaming_fingerprint_matches_saved_codec_with_unicode_and_empty_inputs(self):
        for records in ([], [
                ({"id": "p0", "path_id": "x"}, "first\nquestion"),
                ({"id": "p1", "path_id": "x"}, "\u5317\u4eac \"second\"")]):
            expected = signature({
                "items": [r[0] for r in records],
                "texts": [r[1] for r in records], "digest": DIGEST})
            self.assertEqual(input_fingerprint(DIGEST, lambda: iter(records)), expected)

    def test_complete_rows_pass_but_reordered_ids_and_changed_text_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "indexes").mkdir()
            records = [({"id": "a"}, "alpha"), ({"id": "b"}, "beta")]
            values = np.zeros((2, 1024), dtype=np.float32)
            values[:, 0] = 1
            np.save(run / "indexes/entities.npy", values)
            meta = {
                "items": [r[0] for r in records], "model_digest": DIGEST,
                "dimensions": 1024, "normalized": True,
                "fingerprint": input_fingerprint(DIGEST, lambda: iter(records))}
            write_json(run / "indexes/entities.json", meta)
            self.assertEqual(audit_one(
                run, "entities", lambda: iter(records), 2, DIGEST)["status"], "passed")
            meta["items"].reverse()
            write_json(run / "indexes/entities.json", meta)
            with self.assertRaisesRegex(ValueError, "ID/order"):
                audit_one(run, "entities", lambda: iter(records), 2, DIGEST)
            meta["items"].reverse()
            write_json(run / "indexes/entities.json", meta)
            records[1] = ({"id": "b"}, "changed text")
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                audit_one(run, "entities", lambda: iter(records), 2, DIGEST)


if __name__ == "__main__":
    unittest.main()
