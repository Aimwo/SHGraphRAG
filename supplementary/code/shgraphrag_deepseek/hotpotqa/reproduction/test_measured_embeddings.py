"""Offline regression tests for the observed local embedding runner failure."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from ollama import ResponseError

from config.settings import EMBEDDING_MAX_IN_FLIGHT
from model.measured_embeddings import MeasuredEmbeddings
from reproduction.runtime import ROOT, completed_interval_peak


RUNNER_FAILURE = (
    'Post "http://127.0.0.1:60294/tokenize": dial tcp 127.0.0.1:60294: '
    "connectex: No connection could be made because the target machine "
    "actively refused it."
)


class EmbeddingRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(
            prefix="offline_embedding_", dir=ROOT / "reproduction/runs"
        )
        self.root = Path(self.temporary.name)
        self.environment = patch.dict(os.environ, {"SHGRAPH_RUN_ROOT": str(self.root)})
        self.environment.start()
        self.gate_root = patch("model.measured_embeddings.ROOT", self.root)
        self.gate_root.start()

    def tearDown(self):
        self.gate_root.stop()
        self.environment.stop()
        self.temporary.cleanup()

    def records(self):
        path = self.root / "embedding_calls.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()]

    def test_distinct_wrappers_share_four_slots_and_preserve_vectors(self):
        state = {"active": 0, "peak": 0}
        lock = threading.Lock()
        first_four_entered = threading.Event()
        fifth_entered = threading.Event()
        release = threading.Event()

        class Delegate:
            model = "bge-m3:latest"
            base_url = "http://127.0.0.1:11434"

            def embed_documents(self, texts):
                with lock:
                    state["active"] += 1
                    state["peak"] = max(state["peak"], state["active"])
                    if state["active"] >= 4:
                        first_four_entered.set()
                    if state["active"] > 4:
                        fifth_entered.set()
                try:
                    if not release.wait(timeout=5):
                        raise TimeoutError("Offline delegate was not released")
                    return [[float(len(text)), 0.5] for text in texts]
                finally:
                    with lock:
                        state["active"] -= 1

        models = [MeasuredEmbeddings(Delegate()) for _ in range(8)]
        self.assertEqual(EMBEDDING_MAX_IN_FLIGHT, 4)
        self.assertTrue(all(model.slots.capacity == 4 for model in models))
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [
                pool.submit(model.embed_documents, ["a", "abcd"]) for model in models
            ]
            try:
                self.assertTrue(first_four_entered.wait(timeout=3))
                self.assertFalse(fifth_entered.wait(timeout=0.2))
            finally:
                release.set()
            actual = [future.result(timeout=3) for future in futures]
        self.assertEqual(state["peak"], 4)
        self.assertEqual(actual, [[[1.0, 0.5], [4.0, 0.5]]] * 8)
        records = self.records()
        self.assertEqual(len(records), 8)
        self.assertEqual(completed_interval_peak(records), 4)
        self.assertTrue(all(row["local_gate_capacity"] == 4 for row in records))
        self.assertTrue(all(0 <= row["local_gate_slot"] < 4 for row in records))

    def test_interval_peak_counts_overlaps_and_ignores_unfinished_calls(self):
        records = [
            {"sent_unix": 1, "received_unix": 3},
            {"sent_unix": 2, "received_unix": 4},
            {"sent_unix": 3, "received_unix": 5},
            {"sent_unix": 2},
            {"sent_unix": 7, "received_unix": 7},
        ]
        self.assertEqual(completed_interval_peak(records), 2)
        self.assertEqual(completed_interval_peak([]), 0)

    def test_observed_runner_refusal_is_retried_and_every_attempt_is_recorded(self):
        class Delegate:
            model = "bge-m3:latest"
            base_url = "http://127.0.0.1:11434"
            attempts = 0

            def embed_query(self, text):
                self.attempts += 1
                if self.attempts < 3:
                    raise ResponseError(RUNNER_FAILURE, status_code=400)
                return [0.25, 0.75]

        delegate = Delegate()
        with (
            patch("model.measured_embeddings.time.sleep"),
            patch("model.measured_embeddings.time.time",
                  side_effect=[1.0, 1.1, 2.0, 2.1, 3.0, 3.1]),
        ):
            actual = MeasuredEmbeddings(delegate).embed_query("original entity")
        self.assertEqual(actual, [0.25, 0.75])
        self.assertEqual(delegate.attempts, 3)
        records = self.records()
        self.assertEqual([row["status"] for row in records], ["failed", "failed", "completed"])
        self.assertEqual([row["attempt"] for row in records], [1, 2, 3])
        self.assertEqual(len({row["logical_call_id"] for row in records}), 1)
        self.assertEqual(len({tuple(row["text_sha256"]) for row in records}), 1)
        from reproduction.report import make_report
        report = make_report(self.root)
        stage = records[0]["stage"]
        self.assertEqual(report["embedding_calls"], 3)
        self.assertEqual(report["embedding_retry_attempts"], 2)
        self.assertEqual(report["embedding_by_stage"][stage]["failed"], 2)
        self.assertEqual(report["embedding_observed_completed_interval_peak_concurrency"], 1)
        timing = report["embedding_request_latency_by_stage"][stage]
        self.assertEqual(timing["response_seconds"]["count"], 3)
        self.assertEqual(timing["queue_seconds"]["count"], 3)
        self.assertEqual(timing["error_types"], {"ResponseError": 2})

    def test_other_response_errors_are_not_retried(self):
        class Delegate:
            model = "bge-m3:latest"
            base_url = "http://127.0.0.1:11434"
            attempts = 0

            def embed_documents(self, texts):
                self.attempts += 1
                raise ResponseError("input length exceeds context length", status_code=400)

        delegate = Delegate()
        with self.assertRaises(ResponseError):
            MeasuredEmbeddings(delegate).embed_documents(["original candidate"])
        self.assertEqual(delegate.attempts, 1)
        self.assertEqual(len(self.records()), 1)

    def test_runner_retries_are_bounded(self):
        class Delegate:
            model = "bge-m3:latest"
            base_url = "http://127.0.0.1:11434"
            attempts = 0

            def embed_documents(self, texts):
                self.attempts += 1
                raise ResponseError(RUNNER_FAILURE, status_code=400)

        delegate = Delegate()
        with patch("model.measured_embeddings.time.sleep"):
            with self.assertRaises(ResponseError):
                MeasuredEmbeddings(delegate).embed_documents(["original candidate"])
        self.assertEqual(delegate.attempts, 3)
        self.assertEqual(len(self.records()), 3)

    def test_large_document_is_submitted_in_ordered_subbatches_of_32(self):
        calls = []

        class Delegate:
            model = "bge-m3:latest"
            base_url = "http://127.0.0.1:11434"

            def embed_documents(self, texts):
                calls.append(list(texts))
                return [[float(text)] for text in texts]

        texts = [str(index) for index in range(83)]
        actual = MeasuredEmbeddings(Delegate()).embed_documents(texts)
        self.assertEqual([len(batch) for batch in calls], [32, 32, 19])
        self.assertEqual([text for batch in calls for text in batch], texts)
        self.assertEqual(actual, [[float(index)] for index in range(83)])

    def test_observed_socket_buffer_failure_is_retried(self):
        class Delegate:
            model = "bge-m3:latest"
            base_url = "http://127.0.0.1:11434"
            attempts = 0

            def embed_documents(self, texts):
                self.attempts += 1
                if self.attempts == 1:
                    raise ResponseError(
                        'Post "http://127.0.0.1:62833/tokenize": dial tcp '
                        "127.0.0.1:62833: bind: An operation on a socket could "
                        "not be performed because the system lacked sufficient "
                        "buffer space or because a queue was full.",
                        status_code=400,
                    )
                return [[0.1, 0.9]]

        with patch("model.measured_embeddings.time.sleep"):
            actual = MeasuredEmbeddings(Delegate()).embed_documents(["original entity"])
        self.assertEqual(actual, [[0.1, 0.9]])
        self.assertEqual([row["status"] for row in self.records()], ["failed", "completed"])


if __name__ == "__main__":
    unittest.main()
