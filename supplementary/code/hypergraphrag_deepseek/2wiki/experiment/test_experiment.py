"""Offline tests: all HTTP is intercepted; no saved credential is opened."""
from __future__ import annotations

import asyncio
import copy
import importlib.util
import json
import logging
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import numpy as np

from .common import ITEM, PROJECT, ROOT, read_json, read_jsonl, signature, write_json
from .config import load_config, validate
from .evaluate import answer_scores, score_records
from .run import (
    build, corpus_plan, graph_statistics, make_rag, parse_answer, predict_one, validate_prediction,
)
from .transport import ProviderError, Transport, usage_counts
from .verify import audit_requests, usage_ledger
from hypergraphrag.utils import LLMTruncationError

FACTS = (
    '("hyper-relation"<|>"Alice works in Paris."<|>9)##'
    '("entity"<|>"Alice"<|>"person"<|>"Alice works in Paris."<|>90)##'
    '("entity"<|>"Paris"<|>"geo"<|>"Paris is a city."<|>90)<|COMPLETE|>'
)

OFFLINE_GUARDS = []


def setUpModule():
    logging.getLogger("httpx").setLevel(logging.WARNING)
    offline_pause = patch("experiment.transport.require_model_calls_allowed", return_value=None)
    offline_pause.start()
    OFFLINE_GUARDS.append(offline_pause)
    for target in (
            "httpx.AsyncHTTPTransport.handle_async_request",
            "experiment.run.read_authorized_key"):
        guard = patch(target, side_effect=AssertionError(
            "Offline checks must not access a live HTTP transport or saved credential"))
        guard.start()
        OFFLINE_GUARDS.append(guard)


def tearDownModule():
    for guard in reversed(OFFLINE_GUARDS):
        guard.stop()


def configuration():
    config = copy.deepcopy(load_config(ready=False))
    config.update(status="ready", base_url="https://relay.example.invalid/v1",
                  model="deepseek-offline-test",
                  allowed_returned_models=["deepseek-offline-test"])
    return config


def completion(text, *, finish="stop", usage=None, model="deepseek-offline-test"):
    return {"id": "offline-response", "model": model,
            "choices": [{"message": {"role": "assistant", "content": text},
                         "finish_reason": finish}],
            **({"usage": usage} if usage is not None else {})}


def embeddings(body):
    vectors = []
    for text in body["input"]:
        vector = np.zeros(1024, dtype=np.float32)
        vector[0], vector[1] = 1.0, 1.0 + sum(map(ord, text)) % 7 / 10
        vectors.append(vector.tolist())
    return {"model": "bge-m3:latest", "embeddings": vectors, "prompt_eval_count": 7}


class ConfigurationTests(unittest.TestCase):
    def test_pending_destination_rejected_before_secret_or_http(self):
        from . import run
        pending = configuration()
        pending.update(status="awaiting_user_base_url_and_model", base_url=None, model=None)
        with patch.object(run, "load_config", side_effect=lambda: validate(pending)), patch.object(
                run, "read_authorized_key") as secret, patch(
                "httpx.AsyncClient") as http, patch("sys.argv", ["run"]):
            with self.assertRaisesRegex(RuntimeError, "Base URL"):
                run.main()
        secret.assert_not_called()
        http.assert_not_called()

    def test_provider_credential_and_limits_are_bound(self):
        for path, value in (
                (("provider",), "paratera"),
                (("credential_id",), "different-key"),
                (("max_in_flight",), 64),
                (("embedding", "base_url"), "https://external.example"),
                (("embedding", "max_in_flight"), 32),
                (("sample_seed",), 20261020)):
            config = configuration()
            target = config
            for field in path[:-1]:
                target = target[field]
            target[path[-1]] = value
            with self.subTest(path=path), self.assertRaises(ValueError):
                validate(config)

    def test_unknown_token_usage_does_not_become_zero(self):
        self.assertIsNone(usage_counts(None)["input_tokens"])
        result = usage_ledger([{"raw_usage": None},
                               {"raw_usage": {"prompt_tokens": 5, "completion_tokens": 0}}])
        self.assertEqual(result["input_tokens"]["known_sum"], 5)
        self.assertEqual(result["input_tokens"]["attempts_without_value"], 1)
        self.assertEqual(result["output_tokens"]["known_sum"], 0)
        self.assertIsNone(result["reasoning_output_tokens"]["known_sum"])


class TransportTests(unittest.IsolatedAsyncioTestCase):
    def transport(self, root, text, embed=None):
        return Transport(
            configuration(), root / "cache", root, key="offline-dummy-key",
            text_client=httpx.AsyncClient(transport=httpx.MockTransport(text), trust_env=False),
            embedding_client=httpx.AsyncClient(
                transport=httpx.MockTransport(embed or (lambda request: httpx.Response(
                    200, json=embeddings(json.loads(request.content))))), trust_env=False))

    async def test_exact_prompt_usage_cache_and_redaction(self):
        requests = []

        def respond(request):
            requests.append(request)
            return httpx.Response(200, json=completion(
                "ok", usage={"prompt_tokens": 10, "completion_tokens": 3,
                             "prompt_tokens_details": {"cached_tokens": 2}}))

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            transport = self.transport(root, respond)
            try:
                for _ in range(2):
                    result = await transport.complete(
                        "original question", system_prompt="original system",
                        history_messages=[{"role": "assistant", "content": "original history"}])
                    self.assertEqual(result, "ok")
                self.assertEqual(len(requests), 1)
                payload = json.loads(requests[0].content)
                self.assertEqual(payload["messages"], [
                    {"role": "system", "content": "original system"},
                    {"role": "assistant", "content": "original history"},
                    {"role": "user", "content": "original question"}])
                self.assertNotIn("temperature", payload)
                self.assertNotIn("max_tokens", payload)
                row = read_jsonl(root / "api_calls.jsonl")[0]
                self.assertEqual(row["raw_usage"]["prompt_tokens"], 10)
                self.assertEqual(row["usage"]["cached_input_tokens"], 2)
                artifact = read_json(root / row["raw_artifact"])
                self.assertEqual(artifact["request"], payload)
                self.assertIn("original question", json.dumps(artifact))
                self.assertNotIn("offline-dummy-key", json.dumps(artifact))
                self.assertEqual(transport.hits, 1)
            finally:
                await transport.close()

    async def test_retry_retains_failed_usage_and_identical_request(self):
        requests = []

        def respond(request):
            requests.append(json.loads(request.content))
            if len(requests) == 1:
                return httpx.Response(500, json={"error": {"code": 500},
                                               "usage": {"prompt_tokens": 2}})
            return httpx.Response(200, json=completion("ok", usage={"prompt_tokens": 3}))

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            transport = self.transport(root, respond)
            try:
                with patch("experiment.transport.asyncio.sleep", new=AsyncMock()):
                    self.assertEqual(await transport.complete("same input"), "ok")
                self.assertEqual(requests[0], requests[1])
                records = read_jsonl(root / "api_calls.jsonl")
                self.assertEqual([r["status"] for r in records], ["failed", "completed"])
                self.assertEqual(usage_ledger(records)["input_tokens"]["known_sum"], 5)
                self.assertEqual(len(list((root / "cache/attempts").glob("*.json"))), 2)
            finally:
                await transport.close()

    async def test_wrong_returned_model_is_terminal_and_audited(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            transport = self.transport(root, lambda r: httpx.Response(
                200, json=completion("wrong", model="another-provider-model")))
            try:
                with self.assertRaisesRegex(ProviderError, "returned_model"):
                    await transport.complete("original")
                self.assertEqual(transport.calls, 1)
                row = read_jsonl(root / "api_calls.jsonl")[0]
                self.assertEqual(row["returned_model"], "another-provider-model")
                self.assertEqual(row["status"], "failed")
            finally:
                await transport.close()

    async def test_attempt_payload_tampering_and_missing_final_evidence_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            transport = self.transport(root, lambda r: httpx.Response(200, json=completion("ok")))
            try:
                await transport.complete("original input")
                self.assertEqual(len(audit_requests(root, configuration())["text"]), 1)
                row = read_jsonl(root / "api_calls.jsonl")[0]
                path = root / row["raw_artifact"]
                original = read_json(path)
                altered = copy.deepcopy(original)
                altered["request"]["messages"][-1]["content"] = "different input"
                write_json(path, altered)
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    audit_requests(root, configuration())
                write_json(path, original)
                (root / "api_calls.jsonl").write_text("", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "lacks final evidence"):
                    audit_requests(root, configuration())
            finally:
                await transport.close()

    async def test_embedding_order_cache_batching_and_no_authorization_header(self):
        requests = []

        def embed(request):
            self.assertEqual(str(request.url), "http://127.0.0.1:11434/api/embed")
            self.assertNotIn("authorization", request.headers)
            body = json.loads(request.content)
            self.assertNotIn("truncate", body)
            self.assertEqual(body["options"], {"num_batch": 8192})
            self.assertLessEqual(len(body["input"]), 32)
            requests.append(body)
            return httpx.Response(200, json=embeddings(body))

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            transport = self.transport(root, lambda r: httpx.Response(500), embed)
            try:
                texts = [f"text-{i}" for i in range(35)] + ["text-1"]
                output = await transport.embed(texts)
                self.assertEqual([len(r["input"]) for r in requests], [32, 3])
                np.testing.assert_array_equal(output[1], output[-1])
                cached = await transport.embed(list(reversed(texts)))
                np.testing.assert_array_equal(cached, output[::-1])
                self.assertEqual(len(requests), 2)
                self.assertEqual(output.shape, (36, 1024))
            finally:
                await transport.close()

    async def test_combined_limit100_and_embedding_limit4(self):
        active = {"all": 0, "embed": 0, "peak_all": 0, "peak_embed": 0}

        async def respond(request):
            local = request.url.host == "127.0.0.1"
            active["all"] += 1
            active["embed"] += int(local)
            active["peak_all"] = max(active["peak_all"], active["all"])
            active["peak_embed"] = max(active["peak_embed"], active["embed"])
            try:
                await asyncio.sleep(0.01)
                data = json.loads(request.content)
                return httpx.Response(200, json=embeddings(data) if local else completion("ok"))
            finally:
                active["all"] -= 1
                active["embed"] -= int(local)

        with tempfile.TemporaryDirectory() as folder:
            transport = self.transport(Path(folder), respond, respond)
            try:
                await asyncio.gather(
                    *(transport.embed([f"embedding-{i}"]) for i in range(10)),
                    *(transport.complete(f"question-{i}") for i in range(145)))
                self.assertLessEqual(active["peak_all"], 100)
                self.assertGreater(active["peak_embed"], 1)
                self.assertLessEqual(active["peak_embed"], 4)
                self.assertEqual(transport.peak_requests, 100)
                self.assertEqual(active["all"], 0)
            finally:
                await transport.close()

    async def test_truncated_extraction_never_becomes_partial_graph_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            transport = self.transport(root, lambda r: httpx.Response(
                200, json=completion(FACTS, finish="length")))
            try:
                rag = make_rag(root / "index", transport)
                await rag.ainsert(["Alice works in Paris."])
                stats = graph_statistics(rag)
                # Native ainsert returns before committing doc/chunk KV stores
                # when every extraction is unavailable. Full-run build rejects it.
                self.assertEqual(stats["full_docs"], 0)
                self.assertEqual(stats["chunks"], 0)
                self.assertEqual(stats["entity_nodes"], 0)
                self.assertEqual(stats["hyperedge_nodes"], 0)
                first = read_jsonl(root / "api_calls.jsonl")[0]
                request = read_json(root / first["raw_artifact"])["request"]
                calls = transport.calls
                with self.assertRaises(LLMTruncationError):
                    await transport.complete(request["messages"][-1]["content"])
                self.assertEqual(transport.calls, calls)
                self.assertEqual(transport.hits, 1)
            finally:
                await transport.close()

    async def test_native_hypergraph_build_retrieval_and_short_answer(self):
        requests = []

        def respond(request):
            body = json.loads(request.content)
            requests.append(body)
            text = ('{"answer":"Paris"}' if any(
                m["role"] == "system" for m in body["messages"]) else FACTS)
            return httpx.Response(200, json=completion(text, usage={"prompt_tokens": 5,
                                                                  "completion_tokens": 2}))

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            transport = self.transport(root, respond)
            try:
                corpus = [{"id": "offline-passage", "text": "Alice\nAlice works in Paris."}]
                mapping, _ = corpus_plan(corpus)
                rag = make_rag(root / "index", transport)
                await rag.ainsert([corpus[0]["text"]])
                stats = graph_statistics(rag)
                self.assertEqual(stats["entity_nodes"], 2)
                self.assertEqual(stats["hyperedge_nodes"], 1)
                result = await predict_one(
                    {"id": "2hop__offline", "question": "Where does Alice work?"},
                    rag, transport, mapping, root,
                    provenance_sha256="offline-contract", index_sha256="offline-index")
                self.assertEqual(result["answer"], "Paris")
                self.assertEqual(result["retrieved_passage_ids"], ["offline-passage"])
                self.assertEqual(result["status"], "completed")
                self.assertTrue(result["context"])
                self.assertTrue((root / "predictions/2hop__offline.json").exists())
                self.assertEqual({r["model"] for r in requests}, {"deepseek-offline-test"})
            finally:
                await transport.close()

    async def test_completed_index_is_reused_and_changed_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            transport = self.transport(root, lambda r: httpx.Response(200, json=completion(FACTS)))
            try:
                corpus = [{"id": "offline-passage", "text": "Alice\nAlice works in Paris."}]
                _, _, index = await build(root, transport, corpus, {})
                before = transport.calls, transport.embedding_calls
                await build(root, transport, corpus, {})
                self.assertEqual(before, (transport.calls, transport.embedding_calls))
                path = root / index["directory"] / "kv_store_full_docs.json"
                path.write_bytes(path.read_bytes() + b"\n")
                with self.assertRaisesRegex(ValueError, "file hashes changed"):
                    await build(root, transport, corpus, {})
            finally:
                await transport.close()

    async def test_failed_and_fatal_queries_keep_timing_and_immutable_attempts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            transport = self.transport(root, lambda r: httpx.Response(500))
            query = {"id": "2hop__offline", "question": "Where?"}
            try:
                rag = SimpleNamespace(aquery=AsyncMock(side_effect=ProviderError("http_503")))
                result = await predict_one(
                    query, rag, transport, {}, root,
                    provenance_sha256="offline-contract", index_sha256="offline-index")
                self.assertEqual(result["status"], "failed")
                self.assertEqual(result["error"]["stage"], "retrieval")
                self.assertGreaterEqual(result["retrieval_s"], 0)
                rag.aquery = AsyncMock(side_effect=ProviderError("http_401"))
                with self.assertRaisesRegex(ProviderError, "http_401"):
                    await predict_one(
                        query, rag, transport, {}, root,
                        provenance_sha256="offline-contract", index_sha256="offline-index")
                attempts = read_jsonl(root / "query_attempts.jsonl")
                self.assertEqual(len(attempts), 2)
                self.assertTrue(all((root / a["artifact"]).exists() for a in attempts))
                self.assertEqual(read_json(root / "predictions/2hop__offline.json")["status"], "failed")
                rag.aquery = AsyncMock(side_effect=LLMTruncationError(
                    "output_truncated", "offline-request", "partial extraction", "offline-response"))
                result = await predict_one(
                    query, rag, transport, {}, root,
                    provenance_sha256="offline-contract", index_sha256="offline-index")
                self.assertEqual(result["error"]["stage"], "retrieval")
                self.assertEqual(result["raw_answer"], "")
                self.assertEqual(result["unavailable_raw_text"], "partial extraction")
            finally:
                await transport.close()


class EvaluationTests(unittest.TestCase):
    def test_resume_rejects_changed_context_or_model(self):
        query = {"id": "2hop__offline", "question": "Where?"}
        result = {
            **query, "status": "completed", "answer": "Paris", "context": "original evidence",
            "context_sha256": signature("original evidence"),
            "provenance_sha256": "contract", "index_sha256": "index",
            "provider": "deepseek-relay", "model": configuration()["model"],
            "credential_id": configuration()["credential_id"], "retrieved_passage_ids": ["passage"]}
        validate_prediction(result, query, configuration(), "contract", "index", {"passage"})
        for key, value in (("context", "different evidence"), ("model", "different-model")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_prediction({**result, key: value}, query, configuration(),
                                    "contract", "index", {"passage"})

    def test_answer_metrics_match_supplied_musique_answer_metric(self):
        path = ROOT.parents[2] / "code/shgraphrag_gpt4omini/qa/musique_evaluate_v1.0.py"
        spec = importlib.util.spec_from_file_location("reference_musique_metric", path)
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        cases = [
            ("The Eiffel Tower!", "eiffel tower"),
            ("New-York", "New York"), ("", ""), ("The", ""),
            ("City of Paris", "Paris"), ("42", "43"), ("Paris Paris", "Paris"),
        ]
        for prediction, gold in cases:
            with self.subTest(prediction=prediction, gold=gold):
                em, f1 = answer_scores(prediction, gold)
                self.assertEqual(em, reference.compute_exact(gold, prediction))
                self.assertAlmostEqual(f1, reference.compute_f1(gold, prediction))

    def test_missing_and_failed_count_in_fixed1000_denominator(self):
        ids = [f"2hop__{i}" for i in range(1000)]
        gold = [{"id": qid, "answer": "Paris", "answer_aliases": ["City of Paris"],
                 "paragraphs": [{"title": "Paris", "paragraph_text": "Paris is a city.",
                                 "is_supporting": True}]} for qid in ids]
        predictions = {
            ids[0]: {"id": ids[0], "status": "completed", "answer": "City of Paris"},
            ids[1]: {"id": ids[1], "status": "failed", "answer": "Paris"}}
        result = score_records(ids, gold, predictions)
        self.assertEqual(result["denominator"], 1000)
        self.assertEqual(result["native_alias_em"], 0.1)
        self.assertEqual(result["missing_or_failed"], 999)
        self.assertEqual(result["per_sample"][1]["native_alias_em"], 0)
        predictions["extra"] = {}
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            score_records(ids, gold, predictions)

    def test_normalization_and_answer_schema(self):
        self.assertEqual(answer_scores("The Eiffel Tower!", "eiffel tower"), (1.0, 1.0))
        self.assertEqual(parse_answer('```json\n{"answer":"Paris"}\n```'), "Paris")
        with self.assertRaises(ValueError):
            parse_answer('{"answer":"Paris","extra":"not allowed"}')


if __name__ == "__main__":
    unittest.main()
