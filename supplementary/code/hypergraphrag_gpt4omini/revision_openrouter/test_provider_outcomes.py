"""Regression checks for real provider refusals without lost corpus or IDs."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
import numpy as np

from hypergraphrag import HyperGraphRAG
from hypergraphrag.operate import _handle_entity_relation_summary
from hypergraphrag.utils import (
    EmbeddingFunc, LLMRefusalError, LLMTruncationError, LLMUnavailableError,
)
from .run import LOCK, read_json, run_dataset
from .efficiency import ledger
from .test_revision import TEST_ENV, completion
from .transport import ITEM, Transport
from .verify import validate_response


FACTS = (
    '("hyper-relation"<|>"Alice works in Paris."<|>9)##'
    '("entity"<|>"Alice"<|>"person"<|>"Alice works in Paris."<|>90)##'
    '("entity"<|>"Paris"<|>"geo"<|>"Paris is a city."<|>90)<|COMPLETE|>')


async def vectors(texts):
    output = np.ones((len(texts), 1024), dtype=np.float32)
    return output / np.linalg.norm(output, axis=1, keepdims=True)


class ProviderOutcomeTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_finish_is_unavailable_without_retry_or_partial_facts(self):
        for omit_finish in (False, True):
            with self.subTest(omit_finish=omit_finish), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with patch.dict("os.environ", TEST_ENV):
                    transport = Transport(root / "cache", root / "logs")
                await transport.text_client.aclose()
                payload = completion(FACTS, finish=None)
                if omit_finish:
                    del payload["choices"][0]["finish_reason"]
                requests = []
                def respond(request):
                    requests.append(request)
                    return httpx.Response(200, json=payload)
                transport.text_client = httpx.AsyncClient(
                    transport=httpx.MockTransport(respond))
                token = ITEM.set({"dataset": "2wiki"})
                try:
                    for _ in range(2):
                        with self.assertRaises(LLMUnavailableError) as caught:
                            await transport.complete("-Goal- original extraction", max_tokens=8192)
                        self.assertEqual(caught.exception.outcome, "output_incomplete")
                    self.assertEqual(len(requests), 1)
                    self.assertEqual(transport.calls, 0)
                    self.assertEqual(transport.filtered_calls, 0)
                    self.assertEqual(transport.truncated_calls, 0)
                    self.assertEqual(transport.incomplete_calls, 1)
                    self.assertEqual(transport.incomplete_cache_hits, 1)
                    saved = read_json(next((root / "cache/responses").glob("*.json")))
                    self.assertEqual(saved["status"], "output_incomplete")
                    self.assertEqual(saved["raw_response"], payload)
                    validate_response(saved, saved["request_sha256"])
                    altered = {**saved, "raw_response": completion(FACTS, finish="stop")}
                    with self.assertRaises(RuntimeError):
                        validate_response(altered, saved["request_sha256"])
                    calls = [json.loads(line) for line in
                             (root / "logs/api_calls.jsonl").read_text().splitlines()]
                    hits = [json.loads(line) for line in
                            (root / "logs/cache_hits.jsonl").read_text().splitlines()]
                    totals = ledger(calls, hits)
                    self.assertEqual(totals["attempts"], 1)
                    self.assertEqual(totals["incomplete_output_responses"], 1)
                    self.assertEqual(totals["local_incomplete_response_cache_hits"], 1)
                    self.assertEqual(totals["input_tokens"], 4)
                finally:
                    ITEM.reset(token)
                    await transport.close()

    async def test_incomplete_initial_output_keeps_corpus_without_using_partial_facts(self):
        from hypergraphrag.utils import LLMIncompleteError
        async def complete(prompt, **kwargs):
            if "IncompleteDoc" in prompt:
                raise LLMIncompleteError("completion_missing_finish_reason", "initial", FACTS)
            if "MANY knowdge" in prompt:
                raise LLMIncompleteError("completion_missing_finish_reason", "glean", FACTS)
            return FACTS
        with tempfile.TemporaryDirectory() as directory:
            rag = HyperGraphRAG(
                working_dir=directory, enable_llm_cache=False,
                llm_model_func=complete, llm_model_max_async=2,
                embedding_func=EmbeddingFunc(1024, 8192, vectors), log_level="ERROR")
            await rag.ainsert(["IncompleteDoc\nPartial output.", "GoodDoc\nAlice works in Paris."])
            self.assertEqual(len(rag.full_docs._data), 2)
            self.assertEqual(len(rag.text_chunks._data), 2)
            good = {key for key, value in rag.text_chunks._data.items()
                    if "GoodDoc" in value["content"]}
            graph = rag.chunk_entity_relation_graph._graph
            self.assertEqual(graph.number_of_nodes(), 3)
            self.assertEqual({data["source_id"] for _, data in graph.nodes(data=True)}, good)

    async def test_explicit_filter_and_refusal_are_terminal_and_replayable(self):
        payloads = [completion("partial untrusted text", finish="content_filter"),
                    completion(None), completion("partial output", finish="length")]
        payloads[1]["choices"][0]["message"]["refusal"] = "Provider refusal"
        for payload in payloads:
            with self.subTest(finish=payload["choices"][0]["finish_reason"]), \
                    tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with patch.dict("os.environ", TEST_ENV):
                    transport = Transport(root / "cache", root / "logs")
                await transport.text_client.aclose()
                requests = []
                def respond(request):
                    requests.append(request)
                    return httpx.Response(200, json=payload)
                transport.text_client = httpx.AsyncClient(
                    transport=httpx.MockTransport(respond))
                try:
                    for _ in range(2):
                        with self.assertRaises(LLMUnavailableError):
                            await transport.complete("Exact original request", max_tokens=64)
                    self.assertEqual(len(requests), 1)
                    self.assertEqual(transport.calls, 0)
                    truncated = payload["choices"][0]["finish_reason"] == "length"
                    self.assertEqual(transport.filtered_calls, int(not truncated))
                    self.assertEqual(transport.filtered_cache_hits, int(not truncated))
                    self.assertEqual(transport.truncated_calls, int(truncated))
                    self.assertEqual(transport.truncated_cache_hits, int(truncated))
                    saved = read_json(next((root / "cache/responses").glob("*.json")))
                    self.assertEqual(saved["status"], "output_truncated" if truncated else "provider_filtered")
                    self.assertEqual(saved["raw_response"], payload)
                    validate_response(saved, saved["request_sha256"])
                finally:
                    await transport.close()

    async def test_filtered_initial_and_glean_keep_documents_and_prior_facts(self):
        requests = []
        async def complete(prompt, history_messages=None, **kwargs):
            requests.append(prompt)
            if "DeniedDoc" in prompt:
                raise LLMRefusalError("completion_content_filter", "initial")
            if "MANY knowdge" in prompt:
                raise LLMRefusalError("completion_content_filter", "glean")
            return FACTS
        with tempfile.TemporaryDirectory() as directory:
            rag = HyperGraphRAG(
                working_dir=directory, enable_llm_cache=False,
                llm_model_func=complete, llm_model_max_async=2,
                embedding_func=EmbeddingFunc(1024, 8192, vectors), log_level="ERROR")
            await rag.ainsert(["DeniedDoc\nUnavailable extraction.", "GoodDoc\nAlice works in Paris."])
            self.assertEqual(len(rag.full_docs._data), 2)
            self.assertEqual(len(rag.text_chunks._data), 2)
            graph = rag.chunk_entity_relation_graph._graph
            self.assertGreater(graph.number_of_nodes(), 0)
            sources = {data["source_id"] for _, data in graph.nodes(data=True)}
            good = {key for key, value in rag.text_chunks._data.items()
                    if "GoodDoc" in value["content"]}
            self.assertEqual(sources, good)
            self.assertEqual(len(requests), 3)
            self.assertFalse(any("whether knowdge" in prompt for prompt in requests))

    async def test_filtered_summary_retains_existing_evidence(self):
        description = "Original description. " * 100
        async def refused(*args, **kwargs):
            raise LLMRefusalError("completion_content_filter", "summary")
        result = await _handle_entity_relation_summary("Alice", description, {
            "llm_model_func": refused, "llm_model_max_token_size": 8192,
            "tiktoken_model_name": "gpt-4o-mini", "entity_summary_to_max_tokens": 10,
            "addon_params": {},
        })
        self.assertEqual(result, description)

    async def test_truncated_glean_retains_prior_complete_facts(self):
        async def complete(prompt, **kwargs):
            if "MANY knowdge" in prompt:
                raise LLMTruncationError("completion_length", "glean",
                                         '("entity"<|>"UNFINISHED FACT"')
            return FACTS
        with tempfile.TemporaryDirectory() as directory:
            rag = HyperGraphRAG(
                working_dir=directory, enable_llm_cache=False,
                llm_model_func=complete, llm_model_max_async=2,
                embedding_func=EmbeddingFunc(1024, 8192, vectors), log_level="ERROR")
            await rag.ainsert(["GoodDoc\nAlice works in Paris."])
            graph = rag.chunk_entity_relation_graph._graph
            self.assertEqual(len(rag.full_docs._data), 1)
            self.assertEqual(len(rag.text_chunks._data), 1)
            self.assertEqual(graph.number_of_nodes(), 3)
            self.assertNotIn("UNFINISHED FACT", graph.nodes)

    async def test_filtered_query_or_answer_preserves_frozen_denominator(self):
        entry = read_json(LOCK)["datasets"][0]
        cases = [(stage, finish) for stage in ("query_extract", "answer")
                 for finish in ("content_filter", "length", None)]
        cases.append(("answer", "invalid_schema"))
        for refuse_stage, finish in cases:
            with self.subTest(stage=refuse_stage, finish=finish), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with patch.dict("os.environ", TEST_ENV):
                    transport = Transport(root / "cache", root, workers=2)
                await transport.text_client.aclose()
                await transport.client.aclose()
                def respond(request):
                    body = json.loads(request.content)
                    prompt = body["messages"][-1]["content"]
                    is_query = bool(ITEM.get().get("item_id"))
                    is_answer = body["messages"][0]["role"] == "system"
                    if is_query and (is_answer == (refuse_stage == "answer")):
                        if finish == "invalid_schema":
                            return httpx.Response(200, json=completion("The answer is Paris."))
                        return httpx.Response(200, json=completion(
                            '{"answer": "incomplete', finish=finish))
                    text = ("NO" if "whether knowdge" in prompt else
                            "<|COMPLETE|>" if "MANY knowdge" in prompt else FACTS)
                    return httpx.Response(200, json=completion(text))
                def embed(request):
                    texts = json.loads(request.content)["input"]
                    return httpx.Response(200, json={"embeddings": [[1.0] * 1024 for _ in texts]})
                transport.text_client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
                transport.client = httpx.AsyncClient(transport=httpx.MockTransport(embed))
                token = ITEM.set({})
                try:
                    result = await run_dataset(entry, root, transport,
                                               argparse.Namespace(limit=1, workers=2),
                                               {"status": "running"})
                    self.assertEqual(result["expected"], 1)
                    self.assertEqual(result["completed"], 1)
                    self.assertEqual(result["missing_or_failed"], 0)
                    self.assertEqual(result["primary_em"], 0)
                    prediction = read_json(next((root / entry["dataset"] / "predictions").glob("*.json")))
                    self.assertEqual(prediction["answer"], "")
                    expected = {"content_filter": "provider_filtered", "length": "output_truncated",
                                None: "output_incomplete",
                                "invalid_schema": "invalid_answer_schema"}[finish]
                    self.assertEqual(prediction["answer_outcome"], expected)
                    if finish != "invalid_schema":
                        self.assertTrue(prediction["unavailable_output"]["request_sha256"])
                    if finish == "content_filter":
                        self.assertTrue(prediction["provider_refusal"]["request_sha256"])
                    if refuse_stage == "answer" and finish is None:
                        timing_before = (root / entry["dataset"] / "timing.json").read_bytes()
                        phases_before = (root / "phase_events.jsonl").read_text()
                        with patch("revision_openrouter.run.make_rag",
                                   side_effect=AssertionError("Completed dataset rebuilt")):
                            repeated = await run_dataset(
                                entry, root, transport, argparse.Namespace(limit=1, workers=2),
                                {"status": "running"})
                        self.assertEqual(repeated, result)
                        self.assertEqual(
                            (root / entry["dataset"] / "timing.json").read_bytes(), timing_before)
                        new_phases = (root / "phase_events.jsonl").read_text()[len(phases_before):]
                        self.assertNotIn('"query_phase_start"', new_phases)
                finally:
                    ITEM.reset(token)
                    await transport.close()


if __name__ == "__main__":
    unittest.main()
