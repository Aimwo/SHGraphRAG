"""Real structured provider errors are retried exactly and fully accounted."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from hypergraphrag.utils import LLMUnavailableError
from revision_openrouter.efficiency import ledger
from revision_openrouter.test_revision import TEST_ENV, completion
from revision_openrouter.transport import ProviderError, Transport


def transient_payload():
    payload = completion("UNTRUSTED PARTIAL OUTPUT", finish="error")
    payload["choices"][0]["error"] = {
        "code": 502, "message": "Network connection lost.",
        "metadata": {"error_type": "provider_unavailable"},
    }
    payload["usage"]["cost"] = 0
    return payload


class CompletionErrorTests(unittest.IsolatedAsyncioTestCase):
    async def transport(self, directory, respond):
        root = Path(directory)
        with patch.dict("os.environ", TEST_ENV):
            transport = Transport(root / "cache", root)
        await transport.text_client.aclose()
        transport.text_client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        return root, transport

    async def test_choice_502_retries_exact_payload_and_preserves_failed_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            requests = []
            failed = transient_payload()
            def respond(request):
                requests.append(json.loads(request.content))
                return httpx.Response(200, json=failed if len(requests) == 1 else completion("OK"))
            root, transport = await self.transport(directory, respond)
            try:
                with patch("revision_openrouter.transport.asyncio.sleep", new=AsyncMock()):
                    result = await transport.complete("Original request", max_tokens=8192)
                self.assertEqual(result, "OK")
                self.assertEqual(len(requests), 2)
                self.assertEqual(requests[0], requests[1])
                self.assertEqual(transport.calls, 1)
                records = [json.loads(line) for line in (root / "api_calls.jsonl").read_text().splitlines()]
                self.assertEqual([row["status"] for row in records], ["failed", "completed"])
                self.assertEqual(records[0]["error"], "http_502")
                self.assertEqual(records[0]["provider_error"], failed["choices"][0]["error"])
                self.assertTrue(records[0]["retry_scheduled"])
                self.assertEqual(records[0]["usage_raw"], failed["usage"])
                self.assertEqual(records[0]["request_invocation_id"], records[1]["request_invocation_id"])
                artifact = json.loads((root / records[0]["failure_artifact"]).read_text())
                self.assertEqual(artifact["response"], failed)
                self.assertEqual(artifact["request"], requests[0])
                saved = json.loads(next((root / "cache/responses").glob("*.json")).read_text())
                self.assertEqual(saved["text"], "OK")
                self.assertNotIn("UNTRUSTED", saved["text"])
                totals = ledger(records, [])
                self.assertEqual(totals["attempts"], 2)
                self.assertEqual(totals["input_tokens"], 8)
                self.assertEqual(totals["failed_attempts"], 1)
            finally:
                await transport.close()

    async def test_repeated_choice_502_stops_after_four_identical_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            requests = []
            def respond(request):
                requests.append(json.loads(request.content))
                return httpx.Response(200, json=transient_payload())
            root, transport = await self.transport(directory, respond)
            try:
                with patch("revision_openrouter.transport.asyncio.sleep", new=AsyncMock()):
                    with self.assertRaisesRegex(ProviderError, "http_502"):
                        await transport.complete("Original request", max_tokens=8192)
                self.assertEqual(len(requests), 4)
                self.assertTrue(all(body == requests[0] for body in requests))
                records = [json.loads(line) for line in (root / "api_calls.jsonl").read_text().splitlines()]
                self.assertEqual([row["attempt"] for row in records], [1, 2, 3, 4])
                self.assertEqual([row["retry_scheduled"] for row in records], [True, True, True, False])
                self.assertEqual(len({row["request_invocation_id"] for row in records}), 1)
                self.assertEqual(len(list((root / "cache/text_failures").glob("*.json"))), 4)
                self.assertEqual(list((root / "cache/responses").glob("*.json")), [])
            finally:
                await transport.close()

    async def test_filter_refusal_and_length_are_not_retried_as_transient(self):
        for finish, refusal in (("content_filter", None), ("length", None), ("error", "Refused")):
            with self.subTest(finish=finish, refusal=refusal), tempfile.TemporaryDirectory() as directory:
                requests = []
                payload = transient_payload()
                payload["choices"][0]["finish_reason"] = finish
                payload["choices"][0]["message"]["refusal"] = refusal
                def respond(request):
                    requests.append(request)
                    return httpx.Response(200, json=payload)
                root, transport = await self.transport(directory, respond)
                try:
                    with self.assertRaises(LLMUnavailableError):
                        await transport.complete("Unchanged original request")
                    self.assertEqual(len(requests), 1)
                finally:
                    await transport.close()

    async def test_billing_error_body_is_saved_and_never_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            requests = []
            payload = {"error": {"code": 402, "message": "Credit limit exceeded"}}
            def respond(request):
                requests.append(request)
                return httpx.Response(402, json=payload)
            root, transport = await self.transport(directory, respond)
            try:
                with self.assertRaisesRegex(ProviderError, "http_402"):
                    await transport.complete("Original request")
                self.assertEqual(len(requests), 1)
                row = json.loads((root / "api_calls.jsonl").read_text())
                self.assertFalse(row["retry_scheduled"])
                raw = json.loads((root / row["failure_artifact"]).read_text())
                self.assertEqual(raw["http_status"], 402)
                self.assertEqual(raw["response"], payload)
                self.assertNotIn("test-placeholder", json.dumps(raw))
            finally:
                await transport.close()

    async def test_unstructured_completion_error_remains_fatal(self):
        with tempfile.TemporaryDirectory() as directory:
            requests = []
            payload = completion("Unknown failure", finish="error")
            def respond(request):
                requests.append(request)
                return httpx.Response(200, json=payload)
            root, transport = await self.transport(directory, respond)
            try:
                with self.assertRaisesRegex(ProviderError, "completion_error"):
                    await transport.complete("Original request")
                self.assertEqual(len(requests), 1)
            finally:
                await transport.close()


if __name__ == "__main__":
    unittest.main()
