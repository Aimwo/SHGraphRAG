import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from model.fucheers_responses import FucheersResponsesChat, response_text
from reproduction.runtime import ROOT

SPEC = importlib.util.spec_from_file_location(
    "supplied_musique_eval", ROOT / "qa/musique_evaluate_v1.0.py"
)
MUSIQUE_EVAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MUSIQUE_EVAL)


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        parent = ROOT / "reproduction/runs"
        parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="offline_test_", dir=parent)
        self.root = Path(self.temporary.name).resolve()
        self.assertTrue(self.root.is_relative_to(parent.resolve()))
        self.environment = patch.dict(os.environ, {
            "SHGRAPH_RUN_ROOT": str(self.root),
            "LANGCHAIN_TRACING_V2": "false", "LANGSMITH_TRACING": "false",
        })
        self.environment.start()

    def tearDown(self):
        self.environment.stop()
        self.temporary.cleanup()

    def test_original_messages_and_actual_usage_survive_responses_adapter(self):
        captured = []
        result = {
            "id": "offline-response", "status": "completed", "model": "gpt-6-luna",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "original tuple output"}]}],
            "usage": {
                "input_tokens": 37, "output_tokens": 11, "total_tokens": 48,
                "input_tokens_details": {"cached_tokens": 5},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        }

        def respond(request):
            captured.append(json.loads(request.content))
            return httpx.Response(200, json=result)

        model = FucheersResponsesChat()
        model._client = httpx.Client(transport=httpx.MockTransport(respond))
        with patch("model.fucheers_responses.read_authorized_key", return_value="offline-only-fake"):
            message = model.invoke([SystemMessage(content="Original system\n"), HumanMessage(content="Original passage\n")])
        self.assertEqual(message.content, "original tuple output")
        self.assertEqual(message.usage_metadata["total_tokens"], 48)
        self.assertEqual(captured[0]["input"], [
            {"role": "system", "content": "Original system\n"},
            {"role": "user", "content": "Original passage\n"},
        ])
        self.assertEqual(captured[0]["reasoning"], {"effort": "none"})
        self.assertNotIn("temperature", captured[0])
        record_paths = list((self.root / "requests").glob("*.record.json"))
        self.assertEqual(len(record_paths), 1)
        record = json.loads(record_paths[0].read_text(encoding="utf-8"))
        self.assertEqual(record["usage"], result["usage"])
        self.assertEqual(record["status"], "completed")
        for path in (self.root / "requests").glob("*.json"):
            self.assertNotIn("offline-only-fake", path.read_text(encoding="utf-8"))
        model._client.close()

    def test_http200_incomplete_is_not_success(self):
        with self.assertRaises(RuntimeError):
            response_text({"status": "incomplete", "output": []})

    def test_resume_reuses_exact_completed_merge_response_without_credentials(self):
        from reproduction.runtime import call_context

        result = {
            "id": "offline-merge-response", "status": "completed", "model": "gpt-6-luna",
            "output": [{"type": "message", "content": [
                {"type": "output_text", "text": "[True, False]"}
            ]}],
            "usage": {"input_tokens": 37, "output_tokens": 5, "total_tokens": 42},
        }
        calls = []

        def respond(request):
            calls.append(request)
            return httpx.Response(200, json=result)

        original = FucheersResponsesChat()
        original._client = httpx.Client(transport=httpx.MockTransport(respond))
        with call_context(stage="entity_merge", question_id="frozen-a"):
            with patch("model.fucheers_responses.read_authorized_key", return_value="offline-only-fake"):
                first = original.invoke("original merge prompt")
        resumed = FucheersResponsesChat()
        with call_context(stage="entity_merge", question_id="frozen-a"):
            with patch("model.fucheers_responses.read_authorized_key",
                       side_effect=AssertionError("Cache replay must not load credentials")):
                replay = resumed.invoke("original merge prompt")
        self.assertEqual(replay.content, first.content)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(list((self.root / "requests").glob("*.record.json"))), 1)
        self.assertTrue(replay.response_metadata["cache_replay"])
        self.assertIsNone(replay.usage_metadata)
        self.assertEqual(len((self.root / "merge_response_reuse.jsonl").read_text().splitlines()), 1)
        original._client.close()

    def test_global_extraction_preserves_passage_order_and_cached_reuse(self):
        # Load the actual extractor and actual hash/retry utilities without
        # graph.__init__'s eager database singleton initialization.
        utility_spec = importlib.util.spec_from_file_location(
            "offline_graph_utils", ROOT / "graph/core/utils.py"
        )
        utilities = importlib.util.module_from_spec(utility_spec)
        utility_spec.loader.exec_module(utilities)
        extractor_spec = importlib.util.spec_from_file_location(
            "offline_actual_extractor", ROOT / "graph/extraction/entity_extractor.py"
        )
        extractor_module = importlib.util.module_from_spec(extractor_spec)
        with patch.dict(sys.modules, {"graph.core": utilities}):
            extractor_spec.loader.exec_module(extractor_module)
        EntityRelationExtractor = extractor_module.EntityRelationExtractor

        calls = []

        def respond(prompt):
            text = prompt.to_messages()[-1].content
            calls.append(text)
            if text == "first":
                time.sleep(0.05)
            return AIMessage(content="extracted:" + text)

        extractor = EntityRelationExtractor(
            RunnableLambda(respond), "original system", "{input_text}", [], [],
            cache_dir=str(self.root / "cache"), max_workers=3,
        )
        files = [["doc-a", None, ["first", "second"]], ["doc-b", None, ["second", "third"]]]
        with patch.dict(os.environ, {"SHGRAPH_GLOBAL_EXTRACTION": "1"}):
            actual = extractor.process_chunks(files)
            again = extractor.process_chunks([["doc-c", None, ["third", "first"]]])
        self.assertEqual(actual[0][-1], ["extracted:first", "extracted:second"])
        self.assertEqual(actual[1][-1], ["extracted:second", "extracted:third"])
        self.assertEqual(again[0][-1], ["extracted:third", "extracted:first"])
        self.assertCountEqual(calls, ["first", "second", "third"])

    def test_fatal_provider_response_stops_future_requests(self):
        calls = []

        def respond(request):
            calls.append(request)
            return httpx.Response(401, json={"error": {"code": "invalid_key"}})

        model = FucheersResponsesChat()
        model._client = httpx.Client(transport=httpx.MockTransport(respond))
        with patch("model.fucheers_responses.read_authorized_key", return_value="offline-only-fake"):
            with self.assertRaises(RuntimeError):
                model.invoke("Original question")
            with self.assertRaises(RuntimeError):
                model.invoke("Original question")
        self.assertEqual(len(calls), 1)
        model._client.close()

    def test_cross_process_slots_never_exceed_capacity(self):
        directory = self.root / "gate"
        children = [
            subprocess.Popen(
                [sys.executable, "-m", "reproduction.gate_probe", str(directory)],
                cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            for _ in range(4)
        ]
        events = []
        for child in children:
            stdout, stderr = child.communicate(timeout=30)
            self.assertEqual(child.returncode, 0, stderr)
            for began, ended in json.loads(stdout):
                events.extend([(began, 1), (ended, -1)])
        active = peak = 0
        for _, change in sorted(events):
            active += change
            peak = max(peak, active)
        self.assertEqual(active, 0)
        self.assertLessEqual(peak, 2)
        self.assertEqual(peak, 2)

    def test_musique_aliases_alignment_and_fixed_denominator(self):
        gold = [
            {"id": "a", "answer": "New York City", "answer_aliases": ["NYC"], "answerable": True},
            {"id": "b", "answer": "William Shakespeare", "answer_aliases": [], "answerable": True},
            {"id": "c", "answer": "Paris", "answer_aliases": [], "answerable": True},
        ]
        gold_file = self.root / "gold.jsonl"
        gold_file.write_text("\n".join(json.dumps(row) for row in reversed(gold)), encoding="utf-8")
        pred_file = self.root / "predictions.json"
        pred_file.write_text(json.dumps({"answer": {"b": "Shakespeare", "a": "NYC"}}), encoding="utf-8")
        result = MUSIQUE_EVAL.evaluate(str(pred_file), str(gold_file), ["a", "b", "c"])
        self.assertEqual(result["denominator"], 3)
        self.assertAlmostEqual(result["answer_em"], 1 / 3)
        self.assertAlmostEqual(result["answer_f1"], 5 / 9)
        self.assertEqual(result["missing_prediction_ids"], ["c"])
        pred_file.write_text(json.dumps({"answer": {"unexpected": "Paris"}}), encoding="utf-8")
        with self.assertRaises(ValueError):
            MUSIQUE_EVAL.evaluate(str(pred_file), str(gold_file), ["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
