"""Offline regression checks for shutdown ordering and post-close redaction."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from .common import read_jsonl
from .test_experiment import configuration
from .transport import ProviderError, Transport
from .verify import audit_requests


class RuntimeRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_redaction_after_close_does_not_interleave_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            client = lambda: httpx.AsyncClient(
                transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})))
            transport = Transport(configuration(), root / "cache", root,
                                  key="offline-dummy-key",
                                  text_client=client(), embedding_client=client())
            self.assertEqual(transport.redact("Kursk-Russia"), "Kursk-Russia")
            await transport.close()
            self.assertEqual(transport.redact("cancelled"), "cancelled")
            self.assertEqual(transport.redact({"attempt_id": "abc", "usage": None}),
                             {"attempt_id": "abc", "usage": None})
            self.assertEqual(transport.redact("Kursk-Russia"), "Kursk-Russia")

    async def test_inflight_tasks_are_drained_before_credentials_and_client_close(self):
        from . import run
        sent = asyncio.Event()
        pending = []

        async def slow(request):
            sent.set()
            await asyncio.Event().wait()
            raise AssertionError("A cancelled request must not finish normally")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = configuration()
            transport = Transport(
                config, root / "cache", root, key="offline-dummy-key",
                text_client=httpx.AsyncClient(transport=httpx.MockTransport(slow)),
                embedding_client=httpx.AsyncClient(
                    transport=httpx.MockTransport(lambda request: httpx.Response(200, json={}))))

            async def failed_build(*args):
                pending.append(asyncio.create_task(transport.complete("remaining real stage input")))
                await sent.wait()
                raise ProviderError("http_502")

            with patch("experiment.transport.require_model_calls_allowed", return_value=None), patch.object(
                    run, "validate_inputs", return_value={}), patch.object(
                    run, "store_provenance", return_value="offline-provenance"), patch.object(
                    run, "Transport", return_value=transport), patch.object(
                    run, "read_authorized_key", return_value="offline-dummy-key"), patch.object(
                    run, "read_jsonl", return_value=[]), patch.object(
                    run, "resource_monitor", new=AsyncMock()), patch.object(
                    run, "build", side_effect=failed_build):
                with self.assertRaisesRegex(ProviderError, "http_502"):
                    await run.execute(config, root)
            self.assertTrue(pending[0].done())
            self.assertTrue(pending[0].cancelled())
            attempts = read_jsonl(root / "api_calls.jsonl")
            self.assertEqual(len(attempts), 1)
            self.assertEqual(attempts[0]["status"], "cancelled")
            self.assertNotIn("[REDACTED_KEY]", attempts[0]["attempt_id"])
            self.assertIsNone(attempts[0]["raw_usage"])
            audit_requests(root, config)


if __name__ == "__main__":
    unittest.main()
