"""Check that index transport recovery cannot change research requests."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from reproduction.data import read_jsonl
from reproduction.helpers.local_index_transport import (
    index_transport, recovering_post, socket_resource_error)


URL = "http://127.0.0.1:11434/api/embed"
ERROR = ('do embedding request: Post "http://127.0.0.1:62376/v1/embeddings": '
         'dial tcp 127.0.0.1:62376: bind: An operation on a socket could not be '
         'performed because the system lacked sufficient buffer space or '
         'because a queue was full.')


class Response:
    def __init__(self, status=200, error=""):
        self.status_code, self.error, self.closed = status, error, False
        self.text = error

    def json(self):
        return {"error": self.error}

    def close(self):
        self.closed = True


class LocalIndexTransportTests(unittest.TestCase):
    def run_case(self, responses, callback):
        with tempfile.TemporaryDirectory() as directory:
            now, calls, sleeps = [10.0], [], []
            log = Path(directory) / "transport.jsonl"

            def sleep(seconds):
                sleeps.append(seconds)
                now[0] += seconds

            def operation(url, **kwargs):
                calls.append((now[0], url, kwargs))
                now[0] += 0.25
                return responses[len(calls) - 1]

            call = recovering_post(
                operation, log, clock=lambda: now[0], sleeper=sleep,
                wall_clock=lambda: now[0] + 1000)
            body = {"model": "bge-m3:latest", "input": ["exact full text"],
                    "truncate": False, "keep_alive": "5m"}
            callback(call, body, calls, sleeps, log)

    def test_success_payload_identity_and_spacing(self):
        first, second = Response(), Response()

        def check(call, body, calls, sleeps, log):
            self.assertIs(call(URL, json=body, timeout=(15, 180)), first)
            self.assertIs(call(URL, json=body, timeout=(15, 180)), second)
            self.assertEqual([r[0] for r in calls], [10, 12])
            self.assertEqual(sleeps, [1.75])
            self.assertTrue(all(r[2]["json"] is body for r in calls))
            self.assertEqual(len(read_jsonl(log)), 2)

        self.run_case([first, second], check)

    def test_known_socket_error_retries_identical_body_and_preserves_evidence(self):
        failed, passed = Response(400, ERROR), Response()

        def check(call, body, calls, sleeps, log):
            self.assertIs(call(URL, json=body, timeout=(15, 180)), passed)
            self.assertEqual(len(calls), 2)
            self.assertEqual(sleeps, [10])
            self.assertTrue(failed.closed)
            self.assertTrue(all(r[2]["json"] is body for r in calls))
            records = read_jsonl(log)
            self.assertTrue(records[0]["will_retry"])
            self.assertEqual(records[0]["local_service_error_body"], ERROR)
            self.assertEqual(records[0]["request_sha256"], records[1]["request_sha256"])
            self.assertFalse(records[1]["will_retry"])

        self.run_case([failed, passed], check)

    def test_unknown_http_error_is_returned_once(self):
        failed = Response(400, "Input too long")

        def check(call, body, calls, sleeps, log):
            self.assertIs(call(URL, json=body, timeout=(15, 180)), failed)
            self.assertEqual(len(calls), 1)
            self.assertEqual(sleeps, [])
            self.assertFalse(failed.closed)
            self.assertFalse(read_jsonl(log)[0]["socket_resource_error"])

        self.run_case([failed], check)

    def test_retry_budget_exhaustion_returns_last_failure(self):
        responses = [Response(400, ERROR) for _ in range(4)]

        def check(call, body, calls, sleeps, log):
            self.assertIs(call(URL, json=body, timeout=(15, 180)), responses[-1])
            self.assertEqual(len(calls), 4)
            self.assertEqual(sleeps, [10, 30, 60])
            self.assertFalse(responses[-1].closed)
            self.assertFalse(read_jsonl(log)[-1]["will_retry"])

        self.run_case(responses, check)

    def test_provider_url_cannot_use_this_adapter(self):
        def check(call, body, calls, sleeps, log):
            with self.assertRaisesRegex(ValueError, "restricted"):
                call("https://openrouter.ai/api/v1/chat/completions",
                     json=body, timeout=(15, 180))
            self.assertEqual(calls, [])
            self.assertFalse(log.exists())

        self.run_case([], check)

    def test_query_transport_is_restored_even_after_index_failure(self):
        original = lambda *args, **kwargs: Response()
        embedding = SimpleNamespace(session=SimpleNamespace(post=original))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "index failed"):
                with index_transport(embedding, Path(directory) / "log.jsonl"):
                    self.assertIsNot(embedding.session.post, original)
                    raise RuntimeError("index failed")
        self.assertIs(embedding.session.post, original)

    def test_only_explicit_loopback_socket_resource_errors_match(self):
        self.assertTrue(socket_resource_error(Response(400, ERROR)))
        self.assertFalse(socket_resource_error(Response(500, ERROR)))
        self.assertFalse(socket_resource_error(Response(400, "buffer space")))


if __name__ == "__main__":
    unittest.main()
