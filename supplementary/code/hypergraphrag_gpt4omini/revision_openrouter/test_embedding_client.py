import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from .embedding_client import ENDPOINT, LEGACY_BASE_URL, LocalEmbeddingClient
from .transport import EMBED_DIGEST, EMBED_MODEL, ITEM, file_hash, signature
from .run import CONFIG
from monitoring.verify_full_completion import audit_embedding_http_records


PAYLOAD = {
    "model": EMBED_MODEL, "input": ["unchanged text", "second text"],
    "truncate": False, "keep_alive": "30m",
    "options": {"num_ctx": 8192, "num_batch": 8192},
}


class EmbeddingClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_explicit_context_overflow_uses_audited_native_truncation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bodies = []

            def respond(request):
                body = json.loads(request.content)
                bodies.append(body)
                if body["truncate"] is False:
                    return httpx.Response(
                        400, json={"error": "the input length exceeds the context length"})
                return httpx.Response(200, json={
                    "embeddings": [[1.0] * 1024] * 2, "prompt_eval_count": 8192})

            client = LocalEmbeddingClient(root, transport=httpx.MockTransport(respond))
            try:
                with patch("revision_openrouter.embedding_client.asyncio.sleep",
                           new_callable=AsyncMock):
                    response = await client.post(ENDPOINT, json=PAYLOAD)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(bodies, [PAYLOAD, {**PAYLOAD, "truncate": True}])
                rows = [json.loads(line) for line in
                        (root / "embedding_http_attempts.jsonl").read_text().splitlines()]
                self.assertEqual([row["truncate"] for row in rows], [False, True])
                self.assertFalse(rows[0]["retry_scheduled"])
                artifact = json.loads((root / rows[1]["overflow_artifact"]).read_text())
                self.assertEqual(artifact["original_request"], PAYLOAD)
                self.assertEqual(artifact["fallback_request"], bodies[1])
                self.assertEqual(artifact["response"]["prompt_eval_count"], 8192)
                self.assertEqual(artifact["response_sha256"],
                                 signature(artifact["response"]))
                self.assertEqual(artifact["original_failure_artifact"],
                                 rows[0]["failure_artifact"])
                self.assertEqual(rows[1]["response_sha256"],
                                 artifact["response_sha256"])
                self.assertEqual(rows[1]["overflow_artifact_sha256"],
                                 file_hash(root / rows[1]["overflow_artifact"]))
                self.assertNotEqual(rows[0]["request_sha256"], rows[1]["request_sha256"])
                stats, errors = audit_embedding_http_records(root, None, rows, CONFIG)
                self.assertEqual(stats["fallback_completed"], 1)
                self.assertTrue(all(not values for values in errors.values()), errors)
                artifact["fallback_request"]["keep_alive"] = "1s"
                artifact["fallback_request_sha256"] = signature(artifact["fallback_request"])
                path = root / rows[1]["overflow_artifact"]
                path.write_text(json.dumps(artifact), encoding="utf-8")
                rows[1]["request_sha256"] = artifact["fallback_request_sha256"]
                rows[1]["overflow_artifact_sha256"] = file_hash(path)
                _, errors = audit_embedding_http_records(root, None, rows, CONFIG)
                self.assertTrue(errors["fallback_errors"])
            finally:
                await client.aclose()

    async def test_overflow_fallback_transient_errors_retry_identical_fallback_body(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bodies = []

            def respond(request):
                body = json.loads(request.content)
                bodies.append(body)
                if len(bodies) == 1:
                    return httpx.Response(
                        400, json={"error": "the input length exceeds the context length"})
                if len(bodies) == 2:
                    return httpx.Response(503, json={"error": "temporarily unavailable"})
                return httpx.Response(200, json={"embeddings": [[1.0] * 1024] * 2})

            client = LocalEmbeddingClient(root, transport=httpx.MockTransport(respond))
            try:
                with patch("revision_openrouter.embedding_client.asyncio.sleep",
                           new_callable=AsyncMock):
                    response = await client.post(ENDPOINT, json=PAYLOAD)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(bodies, [
                    PAYLOAD, {**PAYLOAD, "truncate": True}, {**PAYLOAD, "truncate": True}])
                rows = [json.loads(line) for line in
                        (root / "embedding_http_attempts.jsonl").read_text().splitlines()]
                self.assertEqual([row["attempt"] for row in rows], [1, 1, 2])
                self.assertEqual(rows[1]["request_sha256"], rows[2]["request_sha256"])
                self.assertEqual(rows[1]["overflow_artifact"], rows[2]["overflow_artifact"])
                artifact = json.loads((root / rows[-1]["overflow_artifact"]).read_text())
                self.assertEqual(rows[-1]["response_sha256"],
                                 artifact["response_sha256"])
                self.assertEqual(rows[-1]["overflow_artifact_sha256"],
                                 file_hash(root / rows[-1]["overflow_artifact"]))
            finally:
                await client.aclose()

    async def test_transient_400_retries_exact_body_and_keeps_raw_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bodies = []

            def respond(request):
                self.assertEqual(str(request.url), ENDPOINT)
                bodies.append(json.loads(request.content))
                if len(bodies) == 1:
                    return httpx.Response(400, json={"error": "temporary local failure"})
                return httpx.Response(200, json={"embeddings": [[1.0] * 1024] * 2})

            client = LocalEmbeddingClient(
                root, transport=httpx.MockTransport(respond))
            token = ITEM.set({"dataset": "hotpotqa", "item_id": "test-question"})
            try:
                with patch("revision_openrouter.embedding_client.asyncio.sleep",
                           new_callable=AsyncMock):
                    response = await client.post(
                        ENDPOINT, json={k: v for k, v in PAYLOAD.items() if k != "options"})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(bodies, [PAYLOAD, PAYLOAD])
                records = [json.loads(line) for line in
                           (root / "embedding_http_attempts.jsonl").read_text().splitlines()]
                self.assertEqual([r["http_status"] for r in records], [400, 200])
                self.assertEqual([r["attempt"] for r in records], [1, 2])
                self.assertTrue(all(r["request_sha256"] == signature(PAYLOAD) for r in records))
                self.assertTrue(all(r["dataset"] == "hotpotqa" for r in records))
                saved = json.loads((root / records[0]["failure_artifact"]).read_text())
                self.assertEqual(saved["request"], PAYLOAD)
                self.assertEqual(saved["response"], {"error": "temporary local failure"})
                self.assertEqual(records[-1]["digest"], EMBED_DIGEST)
            finally:
                ITEM.reset(token)
                await client.aclose()

    async def test_permanent_400_is_bounded_and_never_cached_as_vector(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            client = LocalEmbeddingClient(root, transport=httpx.MockTransport(
                lambda request: httpx.Response(400, json={"error": "invalid input"})))
            try:
                with patch("revision_openrouter.embedding_client.asyncio.sleep",
                           new_callable=AsyncMock):
                    response = await client.post(ENDPOINT, json=PAYLOAD)
                with self.assertRaises(httpx.HTTPStatusError):
                    response.raise_for_status()
                records = [json.loads(line) for line in
                           (root / "embedding_http_attempts.jsonl").read_text().splitlines()]
                self.assertEqual(len(records), 4)
                self.assertTrue(all(r["status"] == "failed" for r in records))
                self.assertFalse(list(root.rglob("embeddings.sqlite")))
            finally:
                await client.aclose()

    async def test_transport_failure_and_503_retry_but_401_stops(self):
        for mode, expected in (("network", 2), ("503", 2), ("401", 1)):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                count = 0

                def respond(request):
                    nonlocal count
                    count += 1
                    if count == 1:
                        if mode == "network":
                            raise httpx.ReadError("local connection reset", request=request)
                        return httpx.Response(int(mode), json={"error": mode})
                    return httpx.Response(200, json={"embeddings": [[1.0] * 1024] * 2})

                client = LocalEmbeddingClient(
                    Path(temporary), transport=httpx.MockTransport(respond))
                try:
                    with patch("revision_openrouter.embedding_client.asyncio.sleep",
                               new_callable=AsyncMock):
                        response = await client.post(ENDPOINT, json=PAYLOAD)
                    self.assertEqual(count, expected)
                    self.assertEqual(response.status_code, 401 if mode == "401" else 200)
                finally:
                    await client.aclose()

    async def test_cancellation_is_not_retried(self):
        with tempfile.TemporaryDirectory() as temporary:
            count = 0

            async def respond(request):
                nonlocal count
                count += 1
                raise asyncio.CancelledError

            client = LocalEmbeddingClient(
                Path(temporary), transport=httpx.MockTransport(respond))
            try:
                with self.assertRaises(asyncio.CancelledError):
                    await client.post(ENDPOINT, json=PAYLOAD)
                self.assertEqual(count, 1)
            finally:
                await client.aclose()

    async def test_identity_error_fails_before_request(self):
        with tempfile.TemporaryDirectory() as temporary:
            calls = []
            client = LocalEmbeddingClient(Path(temporary), transport=httpx.MockTransport(
                lambda request: calls.append(request) or httpx.Response(200, json={})))
            try:
                for body in ({**PAYLOAD, "truncate": True},
                             {**PAYLOAD, "model": "other-model"},
                             {**PAYLOAD, "options": {"num_ctx": 8192, "num_batch": 2048}}):
                    with self.assertRaises(ValueError):
                        await client.post(ENDPOINT, json=body)
                self.assertEqual(calls, [])
            finally:
                await client.aclose()

    async def test_inherited_origin_routes_to_isolated_service_without_proxy(self):
        with tempfile.TemporaryDirectory() as temporary:
            seen = []

            def respond(request):
                seen.append(str(request.url))
                self.assertNotIn("authorization", request.headers)
                return httpx.Response(200, json={})

            client = LocalEmbeddingClient(
                Path(temporary), transport=httpx.MockTransport(respond))
            try:
                await client.get(LEGACY_BASE_URL + "/api/tags")
                await client.get(LEGACY_BASE_URL + "/api/ps")
                await client.get(LEGACY_BASE_URL + "/api/version")
                await client.post(LEGACY_BASE_URL + "/api/embed", json=PAYLOAD)
                self.assertEqual(seen, [
                    "http://127.0.0.1:11436/api/tags",
                    "http://127.0.0.1:11436/api/ps",
                    "http://127.0.0.1:11436/api/version",
                    ENDPOINT,
                ])
                with self.assertRaises(ValueError):
                    await client.get("http://example.com/api/tags")
            finally:
                await client.aclose()


if __name__ == "__main__":
    unittest.main()
