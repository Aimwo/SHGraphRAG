import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
import numpy as np

from hypergraphrag import HyperGraphRAG, QueryParam
from hypergraphrag.utils import EmbeddingFunc, limit_async_func_call
from revision_openrouter.evaluate import scores
from revision_openrouter.run import campaign_lock, parse_answer
from revision_openrouter.transport import CacheMissError, ITEM, MODEL, ProviderError, Transport


TEST_ENV = {
    "OPENROUTER_API_KEY": "test-placeholder",
    "OPENROUTER_BASE_URL": "https://openrouter.ai/api/v1",
    "OPENROUTER_MODEL": MODEL,
    "OPENROUTER_PROXY": "http://127.0.0.1:6789",
    "OPENROUTER_MAX_CONCURRENCY": "32",
}


def completion(text, model=MODEL, finish="stop"):
    return {
        "id": "test-response", "model": model, "provider": "test-provider",
        "choices": [{"finish_reason": finish,
                     "message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
    }


class MetricTests(unittest.TestCase):
    def test_native_and_primary_are_distinct(self):
        self.assertEqual(scores("yes indeed", "yes"), (0.0, 0.0))
        self.assertAlmostEqual(scores("yes indeed", "yes", hotpot=False)[1], 2 / 3)
        self.assertEqual(scores("The Paris.", "paris"), (1.0, 1.0))
        self.assertEqual(scores("", "", hotpot=False), (1.0, 1.0))

    def test_answer_is_not_guessed_from_prose(self):
        self.assertEqual(parse_answer('{"answer": "Paris"}'), "Paris")
        with self.assertRaises(ValueError):
            parse_answer("The answer is Paris.")
        with self.assertRaises(ValueError):
            parse_answer('{"answer": "Paris", "extra": "unexpected"}')
        for malformed in ("```", "```json\n{\"answer\": \"Paris\"}"):
            with self.assertRaises(ValueError):
                parse_answer(malformed)
        self.assertEqual(parse_answer('```json\n{"answer": "Paris"}\n```'), "Paris")

    def test_duplicate_campaign_cannot_acquire_worker_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            with campaign_lock(Path(directory)):
                with self.assertRaises(OSError):
                    with campaign_lock(Path(directory)):
                        self.fail("Duplicate campaign worker acquired the lock")
            with campaign_lock(Path(directory)):
                pass


class AsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_offline_recovery_replays_exact_cache_and_cannot_submit_a_miss(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.dict("os.environ", TEST_ENV):
                writer = Transport(root / "cache", root / "logs")
            await writer.text_client.aclose()
            writer.text_client = httpx.AsyncClient(transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json=completion("Cached"))))
            self.assertEqual(await writer.complete("Stored", max_tokens=64), "Cached")
            await writer.close()
            with patch.dict("os.environ", {}, clear=True), \
                    patch("revision_openrouter.transport.ProxyTextClient") as remote:
                offline = Transport(root / "cache", root / "logs", cache_only=True)
                try:
                    self.assertEqual(
                        await offline.complete("Stored", max_tokens=64), "Cached")
                    with self.assertRaises(CacheMissError):
                        await offline.complete("Not stored", max_tokens=64)
                    remote.assert_not_called()
                    self.assertIsNone(offline.text_client)
                    self.assertEqual(offline.calls, 0)
                    self.assertEqual(offline.hits, 1)
                    missing = list((root / "logs/cache/missing_text_requests").glob("*.json"))
                    self.assertEqual(len(missing), 1)
                    saved = json.loads(missing[0].read_text())
                    self.assertEqual(saved["request"]["messages"][-1]["content"], "Not stored")
                    self.assertEqual(saved["status"], "cache_miss_no_request_submitted")
                finally:
                    await offline.close()

    async def test_limiter_releases_failed_call(self):
        @limit_async_func_call(1)
        async def work(fail):
            if fail:
                raise RuntimeError("expected")
            return 17
        with self.assertRaises(RuntimeError):
            await work(True)
        self.assertEqual(await asyncio.wait_for(work(False), 1), 17)

    async def test_chat_contract_and_exact_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.dict("os.environ", TEST_ENV):
                transport = Transport(root / "cache", root / "logs")
            await transport.text_client.aclose()
            requests = []
            def respond(request):
                body = json.loads(request.content)
                requests.append(body)
                self.assertEqual(str(request.url), "https://openrouter.ai/api/v1/chat/completions")
                self.assertEqual(body["model"], MODEL)
                self.assertEqual(body["temperature"], 0)
                self.assertEqual(body["max_tokens"], 64)
                self.assertNotIn("reasoning", body)
                self.assertEqual(body["messages"][-1], {"role": "user", "content": "Contract"})
                return httpx.Response(200, json=completion("OK"))
            transport.text_client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
            self.assertEqual(await transport.complete("Contract", max_tokens=64), "OK")
            self.assertEqual(await transport.complete("Contract", max_tokens=64), "OK")
            self.assertEqual(len(requests), 1)
            self.assertEqual(transport.hits, 1)
            artifacts = "".join(p.read_text(encoding="utf-8")
                                for p in root.rglob("*.json"))
            self.assertNotIn("test-placeholder", artifacts)
            await transport.close()

    async def test_wrong_model_is_never_cached_as_success(self):
        for payload in (completion("OK", model="another-model"),):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                with patch.dict("os.environ", TEST_ENV):
                    transport = Transport(root / "cache", root / "logs")
                await transport.text_client.aclose()
                transport.text_client = httpx.AsyncClient(
                    transport=httpx.MockTransport(
                        lambda request: httpx.Response(200, json=payload)))
                with self.assertRaises(ProviderError):
                    await transport.complete("Check", max_tokens=64)
                self.assertEqual(transport.calls, 0)
                self.assertEqual(list((root / "cache/responses").glob("*.json")), [])
                await transport.close()

    async def test_text_proxy_and_loopback_embedding_are_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.dict("os.environ", TEST_ENV):
                transport = Transport(root / "cache", root / "logs")
            await transport.client.aclose()
            await transport.text_client.aclose()
            requests = []
            def local(request):
                requests.append(str(request.url))
                self.assertNotIn("authorization", request.headers)
                return httpx.Response(200, json={"embeddings": [[1.0] * 1024]})
            def text(request):
                self.assertIn("authorization", request.headers)
                return httpx.Response(200, json=completion('{"answer":"Paris"}'))
            transport.client = httpx.AsyncClient(transport=httpx.MockTransport(local))
            transport.text_client = httpx.AsyncClient(transport=httpx.MockTransport(text))
            vector = await transport.embed(["Paris"])
            self.assertEqual(vector.shape, (1, 1024))
            self.assertAlmostEqual(float(np.linalg.norm(vector[0])), 1.0)
            self.assertEqual(await transport.complete("City?", system_prompt="Return JSON."),
                             '{"answer":"Paris"}')
            self.assertEqual(requests, ["http://127.0.0.1:11434/api/embed"])
            await transport.close()

    async def test_all_text_stages_share_one_in_flight_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.dict("os.environ", TEST_ENV):
                transport = Transport(root / "cache", root / "logs", workers=4)
            await transport.text_client.aclose()
            async def respond(request):
                await asyncio.sleep(0.01)
                return httpx.Response(200, json=completion("NO"))
            transport.text_client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
            prompts = [
                "-Goal- unique ", "MANY knowdge unique ", "whether knowdge unique ",
                "comprehensive summary unique ", "Question unique ",
            ]
            await asyncio.gather(*(
                transport.complete(prompts[index % 5] + str(index), system_prompt="Answer.")
                for index in range(40)))
            self.assertEqual(transport.calls, 40)
            self.assertEqual(transport.peak_active_calls, 4)
            self.assertEqual(transport.active_calls, 0)
            self.assertEqual(set(transport.stages), {
                "extract_completed", "glean_completed", "coverage_completed",
                "summarize_completed", "answer_completed"})
            await transport.close()

    async def test_official_hypergraph_and_provenance(self):
        extraction = (
            '("hyper-relation"<|>"Alice works in Paris."<|>9)##'
            '("entity"<|>"Alice"<|>"person"<|>"Alice works in Paris."<|>90)##'
            '("entity"<|>"Paris"<|>"geo"<|>"Paris is a city."<|>90)<|COMPLETE|>')
        async def complete(prompt, **kwargs):
            if "MANY knowdge" in prompt:
                return "<|COMPLETE|>"
            if "whether knowdge" in prompt:
                return "NO"
            return extraction
        async def embed(texts):
            output = np.ones((len(texts), 1024), dtype=np.float32)
            return output / np.linalg.norm(output, axis=1, keepdims=True)
        with tempfile.TemporaryDirectory() as temporary:
            rag = HyperGraphRAG(
                working_dir=temporary, enable_llm_cache=False,
                llm_model_func=complete, llm_model_max_async=1,
                embedding_func=EmbeddingFunc(1024, 8192, embed), log_level="ERROR")
            await rag.ainsert(["Alice\nAlice works in Paris.", "Paris\nParis is a city."])
            graph = rag.chunk_entity_relation_graph._graph
            self.assertEqual(sum(v["role"] == "hyperedge" for _, v in graph.nodes(data=True)), 1)
            self.assertEqual(sum(v["role"] == "entity" for _, v in graph.nodes(data=True)), 2)
            self.assertEqual(graph.number_of_edges(), 2)
            param = QueryParam(only_need_context=True)
            context = await rag.aquery("Where does Alice work?", param)
            self.assertIn("Alice", context)
            self.assertIn("local", param.retrieval_trace)
            self.assertIn("global", param.retrieval_trace)
            chunks = param.retrieval_trace["global"]["chunks"]
            self.assertEqual(len(chunks), 2)
            self.assertTrue(all(c["full_doc_id"].startswith("doc-") for c in chunks))

    async def test_parallel_official_calls_keep_item_and_chunk_provenance(self):
        extraction = (
            '("hyper-relation"<|>"Alice works in Paris."<|>9)##'
            '("entity"<|>"Alice"<|>"person"<|>"Alice works in Paris."<|>90)##'
            '("entity"<|>"Paris"<|>"geo"<|>"Paris is a city."<|>90)<|COMPLETE|>')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict("os.environ", TEST_ENV):
                transport = Transport(root / "cache", root / "logs", workers=2)
            await transport.text_client.aclose()
            async def respond(request):
                body = json.loads(request.content)
                prompt = body["messages"][-1]["content"]
                await asyncio.sleep(0.001)
                text = ("NO" if "whether knowdge" in prompt else
                        "<|COMPLETE|>" if "MANY knowdge" in prompt else extraction)
                return httpx.Response(200, json=completion(text))
            transport.text_client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
            async def embed(texts):
                output = np.ones((len(texts), 1024), dtype=np.float32)
                return output / np.linalg.norm(output, axis=1, keepdims=True)
            async def complete(*args, **kwargs):
                return await transport.complete(*args, **kwargs)
            rag = HyperGraphRAG(
                working_dir=str(root / "index"), enable_llm_cache=False,
                llm_model_func=complete, llm_model_max_async=2,
                embedding_func=EmbeddingFunc(1024, 8192, embed), log_level="ERROR")
            token = ITEM.set({"dataset": "test"})
            try:
                await rag.ainsert(["Alice\nAlice works in Paris.", "Paris\nParis is a city."])
                async def query(qid, question):
                    ITEM.set({"dataset": "test", "item_id": qid, "stage": "query"})
                    return await rag.aquery(question, QueryParam(only_need_context=True))
                await asyncio.gather(query("one", "Where is Alice?"),
                                     query("two", "Where does Alice work?"))
                await query("one", "Where is Alice?")
            finally:
                ITEM.reset(token)
                await transport.close()
            records = [json.loads(line) for line in
                       (root / "logs/api_calls.jsonl").read_text(encoding="utf-8").splitlines()]
            extracted = {r["chunk_id"] for r in records if r["stage"] == "extract"}
            self.assertEqual(extracted, set(rag.text_chunks._data))
            for stage in ("glean", "coverage"):
                self.assertEqual({r["chunk_id"] for r in records if r["stage"] == stage},
                                 extracted)
            for qid in ("one", "two"):
                self.assertEqual(transport.item_counts[("test", qid, "calls")], 1)
                self.assertEqual([r["stage"] for r in records if r.get("item_id") == qid],
                                 ["query_extract"])
            self.assertEqual(transport.item_counts[("test", "one", "hits")], 1)
            hits = [json.loads(line) for line in
                    (root / "logs/cache_hits.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(hits[0]["item_id"], "one")
            self.assertEqual(hits[0]["stage"], "query_extract")


if __name__ == "__main__":
    unittest.main()
