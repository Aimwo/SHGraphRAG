"""Focused safeguards for scoring, corpus boundaries, graph sparsity and API use."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from .client import ProviderError, ResponsesClient, signature, write_json
from .embedding import VectorIndex
from .entity_resolution import (
    candidate_pairs, canonical_mapping, catalog, remap_extractions)
from .evaluate import answer_scores, evaluate_rows, passage_id
from .graph import assemble, entity_key, iter_paths, parse_extraction, parse_json_extraction
from .runner import Runner


ROOT = Path(__file__).resolve().parents[1]


def load_metric(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "qa" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hotpot = load_metric("reference_hotpot", "hotpot_evaluate_v1.py")
        cls.musique = load_metric("reference_musique", "musique_evaluate_v1.0.py")

    def test_hotpot_reference_equivalence(self):
        examples = [("The U.S.", "US"), ("a red car", "red car"),
                    ("yes", "yes indeed"), ("no", "yes"), ("", "a"),
                    ("Greater Accra", "the Greater Accra Region"),
                    ("Alice Alice Bob", "Alice Bob Bob"), ("May 1, 1990", "1990")]
        for prediction, gold in examples:
            em, f1 = answer_scores(prediction, gold)
            self.assertEqual(em, float(self.hotpot.exact_match_score(prediction, gold)))
            self.assertAlmostEqual(f1, self.hotpot.f1_score(prediction, gold)[0])

    def test_musique_reference_equivalence(self):
        for prediction, gold in [("", ""), ("yes indeed", "yes"),
                                 ("New York", "the city of New York")]:
            em, f1 = answer_scores(prediction, gold, hotpot=False)
            self.assertEqual(em, self.musique.compute_exact(gold, prediction))
            self.assertAlmostEqual(f1, self.musique.compute_f1(gold, prediction))

    def gold(self, qid, answer):
        return {"_id": qid, "answer": answer,
                "context": [["Doc", ["text"]]], "supporting_facts": [["Doc", 0]]}

    def test_id_alignment_order_and_missing_denominator(self):
        gold = [self.gold("b", "Bob"), self.gold("a", "Alice")]
        pred = {"a": {"status": "completed", "answer": "Alice"}}
        result = evaluate_rows(gold, pred, "hotpotqa")
        self.assertEqual(result["expected"], 2)
        self.assertEqual(result["completed"], 1)
        self.assertEqual(result["primary_em"], 50)
        self.assertEqual(result["primary_f1"], 50)

    def test_extra_prediction_id_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_rows([self.gold("a", "Alice")], {"outside": {}}, "hotpotqa")

    def test_alias_diagnostic_does_not_change_primary_score(self):
        row = self.gold("a", "British")
        row["answer_id"] = "Q145"
        prediction = {"a": {"status": "completed", "answer": "United Kingdom"}}
        result = evaluate_rows([row], prediction, "2wiki", {
            "Q145": {"aliases": ["United Kingdom"], "demonyms": ["British"]}})
        self.assertEqual(result["primary_em"], 0)
        self.assertEqual(result["native_alias_em"], 100)

class GraphTests(unittest.TestCase):
    def test_json_adapter_preserves_fields_and_rejects_orphan_relation(self):
        value = {"entities": [
            {"name": "Alice", "type": "Person", "description": "A person"},
            {"name": "Paris", "type": "Place", "description": "A city"}],
            "relationships": [{"source": "Alice", "type": "lives in",
                               "target": "Paris", "description": "Alice lives there",
                               "strength": 8}]}
        parsed = parse_json_extraction(value)
        self.assertEqual(parsed["relations"][0]["head"], "alice")
        self.assertEqual(parsed["relations"][0]["tail"], "paris")
        self.assertEqual(parsed["relations"][0]["strength"], 8)
        value["relationships"][0]["target"] = "London"
        with self.assertRaises(ValueError):
            parse_json_extraction(value)

    def test_tuple_schema_endpoints_and_completion(self):
        text = ('("entity"<|>"Alice"<|>"Person"<|>"Alice lives in Paris.")##'
                '("entity"<|>"Paris"<|>"Place"<|>"A city.")##'
                '("relationship"<|>"Alice"<|>"lives in"<|>"Paris"<|>"Alice lives there."<|>8)'
                '<|COMPLETE|>')
        result = parse_extraction(text)
        self.assertEqual(len(result["entities"]), 2)
        self.assertEqual(len(result["relations"]), 1)
        with self.assertRaises(ValueError):
            parse_extraction(text.replace('<|COMPLETE|>', ''))
        with self.assertRaises(ValueError):
            parse_extraction(text.replace('>"Paris"<|>"Place"', '>"London"<|>"Place"'))

    def test_entity_normalization_preserves_identity_distinctions(self):
        self.assertEqual(entity_key(" Alice  Smith "), entity_key("ALICE SMITH"))
        self.assertNotEqual(entity_key("World War I"), entity_key("World War II"))

    def test_bridge_sparsity_preserves_hyperedges_and_two_hop_paths(self):
        def node(name):
            return {"name": name, "type": "Person", "description": name}
        corpus = [{"passage_id": p, "title": p, "text": p, "sentences": [p]}
                  for p in ("doc1", "doc2")]
        def rel(a, b):
            return {"head": a, "tail": b, "type": "knows",
                    "description": a + " knows " + b, "strength": 8}
        extracted = {
            "doc1": {"entities": {n: node(n) for n in ("b", "a", "c", "d", "x", "y")},
                     "relations": [rel("b", "a"), rel("a", "c"), rel("c", "d"),
                                   rel("x", "y")]},
            "doc2": {"entities": {"b": node("b"), "z": node("z")},
                     "relations": [rel("b", "z")]}}
        graph = assemble(corpus, extracted)
        self.assertEqual(graph["stats"]["bridge_entities"], 1)
        self.assertEqual(graph["stats"]["full_triples"], 5)
        self.assertEqual(graph["stats"]["sparse_triples"], 3)
        self.assertEqual(len(graph["hyperedges"]["doc1"]["entity_ids"]), 6)
        paths = list(iter_paths(graph))
        self.assertEqual(len(paths), 2)
        self.assertEqual(len({p["id"] for p in paths}), 2)
        self.assertTrue(all(len(p["triple_ids"]) == 2 for p in paths))

    def test_global_vector_search_and_explicit_allowed_set(self):
        index = VectorIndex([{"id": "other_question_passage"}, {"id": "local_passage"}],
                            np.array([[1., 0.], [0., 1.]], dtype=np.float32))
        vector = np.array([1., 0.], dtype=np.float32)
        self.assertEqual(index.search(vector, 1)[0]["id"], "other_question_passage")
        self.assertEqual(index.search(vector, 1, {"local_passage"})[0]["id"], "local_passage")


class ProviderTests(unittest.TestCase):
    def test_atomic_json_write_retries_windows_sharing_conflict(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "status.json"
            replace = Path.replace
            attempts = []
            def sharing_conflict_once(source, destination):
                attempts.append(source)
                if len(attempts) == 1:
                    raise PermissionError(13, "simulated sharing conflict")
                return replace(source, destination)
            with patch.object(Path, "replace", sharing_conflict_once), \
                    patch("reproduction.client.time.sleep"):
                write_json(target, {"completed": 1000})
            self.assertEqual(len(attempts), 2)
            self.assertEqual(json.loads(target.read_text(encoding="utf-8")), {"completed": 1000})
            self.assertFalse(list(Path(directory).glob("*.tmp")))

    def test_atomic_json_write_stops_on_persistent_access_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "status.json"
            with patch.object(Path, "replace", side_effect=PermissionError(13, "denied")) as replace, \
                    patch("reproduction.client.time.sleep"):
                with self.assertRaises(PermissionError):
                    write_json(target, {"completed": 1000})
            self.assertEqual(replace.call_count, 8)
            self.assertFalse(target.exists())
            self.assertEqual(len(list(Path(directory).glob("*.tmp"))), 1)

    def test_missing_finish_metadata_retries_without_accepting_partial_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-secret-value"}):
                client = ResponsesClient(root, attempts=2)
            class Response:
                status_code = 200
                def __init__(self, finish):
                    self.finish = finish
                def json(self):
                    return {"model": "openai/gpt-4o-mini",
                            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
                            "choices": [{"finish_reason": self.finish,
                                         "message": {"content": "{}"}}]}
            with patch.object(client.session(), "post", side_effect=[
                    Response(None), Response("stop")]) as post, \
                    patch("reproduction.client.time.sleep"):
                result = client.call(stage="test", item_id="one",
                                     system="system", prompt="prompt")
            self.assertEqual(post.call_count, 2)
            self.assertEqual(post.call_args_list[0].kwargs["json"],
                             post.call_args_list[1].kwargs["json"])
            calls = [json.loads(line) for line in (root / "api_calls.jsonl")
                     .read_text(encoding="utf-8").splitlines()]
            self.assertEqual([c["status"] for c in calls], ["failed", "completed"])
            self.assertEqual(calls[0]["error"], "invalid_finish_reason")
            self.assertIsNone(calls[0]["finish_reason"])
            saved = list((root / "cache").rglob("*.json"))
            self.assertEqual(len(saved), 1)
            self.assertEqual(json.loads(saved[0].read_text(encoding="utf-8"))[
                "request_sha256"], result["request_sha256"])

    def test_provider_finish_error_uses_bounded_identical_request_retries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-secret-value"}):
                client = ResponsesClient(root, attempts=2)
            class Response:
                status_code = 200
                def json(self):
                    return {"model": "openai/gpt-4o-mini", "id": "failed-generation",
                            "provider": "Azure",
                            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
                            "choices": [{"finish_reason": "error",
                                         "message": {"content": "incomplete"}}]}
            with patch.object(client.session(), "post", return_value=Response()) as post, \
                    patch("reproduction.client.time.sleep"):
                with self.assertRaises(ProviderError) as caught:
                    client.call(stage="test", item_id="one", system="system", prompt="prompt")
            self.assertEqual(caught.exception.category, "finish_error")
            self.assertEqual(post.call_count, 2)
            self.assertEqual(post.call_args_list[0].kwargs["json"],
                             post.call_args_list[1].kwargs["json"])
            self.assertFalse(list((root / "cache").rglob("*.json")))
            calls = [json.loads(line) for line in (root / "api_calls.jsonl")
                     .read_text(encoding="utf-8").splitlines()]
            self.assertEqual(sum(c["usage"]["completion_tokens"] for c in calls), 4)
            self.assertTrue(all(c["response_id"] == "failed-generation" for c in calls))

    def test_truncation_and_explicit_refusal_do_not_retry(self):
        for finish, refusal, expected in [
                ("length", None, "finish_length"),
                ("error", "I cannot provide that.", "model_refusal"),
                (None, "I cannot provide that.", "model_refusal")]:
            with self.subTest(finish=finish, refusal=bool(refusal)), \
                    tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-secret-value"}):
                    client = ResponsesClient(root, attempts=2)
                class Response:
                    status_code = 200
                    def json(self):
                        return {"model": "openai/gpt-4o-mini", "usage": {},
                                "choices": [{"finish_reason": finish, "message": {
                                    "content": "incomplete", "refusal": refusal}}]}
                with patch.object(client.session(), "post", return_value=Response()) as post:
                    with self.assertRaises(ProviderError) as caught:
                        client.call(stage="test", item_id="one", system="system", prompt="prompt")
                self.assertEqual(caught.exception.category, expected)
                self.assertEqual(post.call_count, 1)

    def test_incomplete_provider_reply_retries_exact_body_and_keeps_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-secret-value"}):
                client = ResponsesClient(root, attempts=2)
            class Response:
                status_code = 200
                def __init__(self, finish):
                    self.finish = finish
                def json(self):
                    return {"model": "openai/gpt-4o-mini",
                            "usage": {"prompt_tokens": 10, "completion_tokens": 2,
                                      "cost": 0.0001},
                            "choices": [{"finish_reason": self.finish,
                                         "message": {"content": "{}"}}]}
            with patch.object(client.session(), "post", side_effect=[
                    Response("content_filter"), Response("stop")]) as post, \
                    patch("reproduction.client.time.sleep"):
                client.call(stage="test", item_id="one", system="system", prompt="prompt")
            bodies = [call.kwargs["json"] for call in post.call_args_list]
            self.assertEqual(bodies[0], bodies[1])
            calls = [json.loads(line) for line in (root / "api_calls.jsonl")
                     .read_text(encoding="utf-8").splitlines()]
            self.assertEqual([c["status"] for c in calls], ["failed", "completed"])
            self.assertEqual(sum(c["usage"]["cost"] for c in calls), 0.0002)

    def test_fixed_endpoint_completion_preserves_all_existing_graph_records(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = Runner.__new__(Runner)
            runner.root = Path(directory)
            original = {
                "entities": [{"name": "Festival", "type": "Event",
                              "description": "The festival is an event."}],
                "relationships": [{"source": "Festival", "type": "is an",
                                   "target": "Event", "description": "An event.",
                                   "strength": 7}]}
            class FakeClient:
                def call(self, **kwargs):
                    records = kwargs["response_schema"]["properties"]["records"]
                    if records["required"] != ["Event"]:
                        raise AssertionError("Missing names are not schema-fixed")
                    return {"request_sha256": "completion", "response_id": "new",
                            "text": json.dumps({"records": {"Event": {
                                "type": "Concept",
                                "description": "An organized occurrence."}}})}
            runner.client = FakeClient()
            parsed, evidence, response = runner.complete_endpoints(
                {"passage_id": "p", "title": "Festival", "text": "The festival."},
                {"text": json.dumps(original), "request_sha256": "base",
                 "response_id": "old"})
            self.assertEqual(parsed["entities"]["festival"], original["entities"][0])
            self.assertEqual(parsed["relations"][0]["tail"], "event")
            self.assertEqual(parsed["relations"][0]["strength"], 7)
            self.assertEqual(evidence["base_request_sha256"], "base")
            self.assertEqual(evidence["required_names"], ["Event"])
            self.assertEqual(response["request_sha256"], "completion")

    def test_external_origin_and_model_overrides_rejected_before_network(self):
        with tempfile.TemporaryDirectory() as directory:
            for override in ({"OPENROUTER_BASE_URL": "https://other.example/v1"},
                             {"OPENROUTER_MODEL": "openai/other-model"}):
                with patch.dict("os.environ", {
                        "OPENROUTER_API_KEY": "test-secret-value", **override}):
                    with self.assertRaises(ProviderError):
                        ResponsesClient(Path(directory))

    def test_wrong_response_model_is_logged_and_not_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-secret-value"}):
                client = ResponsesClient(root, attempts=1)
            class Response:
                status_code = 200
                def json(self):
                    return {"model": "another-model", "usage": {},
                            "choices": [{"finish_reason": "stop",
                                         "message": {"content": "{}"}}]}
            with patch.object(client.session(), "post", return_value=Response()):
                with self.assertRaises(ProviderError):
                    client.call(stage="test", item_id="one",
                                system="system", prompt="prompt")
            self.assertFalse(list((root / "cache").rglob("*.json")))
            log = json.loads((root / "api_calls.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(log["error"], "response_model_mismatch")

    def test_chat_contract_and_cache_never_store_key(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-secret-value"}):
                client = ResponsesClient(root)
            class Response:
                status_code = 200
                def json(self):
                    return {"model": "openai/gpt-4o-mini",
                            "id": "test", "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                            "choices": [{"finish_reason": "stop",
                                         "message": {"content": "OK"}}]}
            with patch.object(client.session(), "post", return_value=Response()) as post:
                first = client.call(stage="test", item_id="one", system="test",
                                    prompt="prompt", max_tokens=64,
                                    response_schema={"type": "object"})
                second = client.call(stage="test", item_id="two", system="test",
                                     prompt="prompt", max_tokens=64,
                                     response_schema={"type": "object"})
                self.assertEqual(post.call_count, 1)
                self.assertNotIn("reasoning", post.call_args.kwargs["json"])
                self.assertNotIn("reasoning_effort", post.call_args.kwargs["json"])
                self.assertNotIn("max_output_tokens", post.call_args.kwargs["json"])
                self.assertEqual(post.call_args.kwargs["json"]["max_tokens"], 64)
                self.assertEqual(post.call_args.args[0],
                                 "https://openrouter.ai/api/v1/chat/completions")
                self.assertFalse(post.call_args.kwargs["allow_redirects"])
                self.assertTrue(post.call_args.kwargs["json"][
                    "response_format"]["json_schema"]["strict"])
                self.assertEqual(first["text"], "OK")
                self.assertTrue(second["cache_hit"])
                self.assertEqual(first["usage"]["input_tokens"], 1)
                self.assertEqual(first["usage"]["output_tokens"], 1)
            for file in root.rglob("*"):
                if file.is_file():
                    self.assertNotIn("test-secret-value", file.read_text(encoding="utf-8"))

class EntityResolutionTests(unittest.TestCase):
    def fixtures(self):
        def entity(name, description):
            return {"name": name, "type": "Person", "description": description}
        return {
            "a": {"entities": {
                "alice": entity("Alice", "Alice is a director."),
                "alice smith": entity("Alice Smith", "Alice Smith directs films."),
                "bob": entity("Bob", "Bob is an actor.")},
                "relations": [
                    {"head": "alice", "tail": "alice smith", "type": "alias",
                     "description": "The same name.", "strength": 8},
                    {"head": "alice smith", "tail": "bob", "type": "directed",
                     "description": "Directed Bob.", "strength": 8}]},
            "b": {"entities": {"alice": entity("Alice", "Alice was born in Paris.")},
                  "relations": []}}

    def test_confirmation_remaps_edges_and_keeps_all_descriptions(self):
        extracted = self.fixtures()
        details, frequencies = catalog(extracted)
        mapping = canonical_mapping(details, frequencies, [("alice", "alice smith")])
        self.assertEqual(mapping["alice smith"], "alice")
        remapped = remap_extractions(extracted, details, mapping)
        self.assertEqual(len(remapped["a"]["relations"]), 1)
        self.assertEqual(remapped["a"]["relations"][0]["head"], "alice")
        corpus = [{"passage_id": pid, "title": pid, "text": pid, "sentences": [pid]}
                  for pid in extracted]
        graph = assemble(corpus, remapped)
        alice = next(e for e in graph["entities"].values() if e["key"] == "alice")
        self.assertEqual(set(alice["aliases"]), {"Alice", "Alice Smith"})
        self.assertEqual(len(alice["descriptions"]), 3)
        self.assertEqual(alice["passage_ids"], ["a", "b"])
        self.assertEqual(graph["stats"]["bridge_entities"], 1)
        rejected = canonical_mapping(details, frequencies, [])
        self.assertNotEqual(rejected["alice"], rejected["alice smith"])

    def test_candidates_use_source_threshold_or_containment_and_local_scope(self):
        extracted = self.fixtures()
        details, _ = catalog(extracted)
        vectors = np.eye(len(details), dtype=np.float32)
        rows = list(candidate_pairs(
            [{"passage_ids": ["a"]}, {"passage_ids": ["b"]}], extracted, details, vectors))
        self.assertEqual([row["keys"] for row in rows], [["alice", "alice smith"]])
        isolated = list(candidate_pairs(
            [{"passage_ids": ["b"]}], extracted, details, vectors))
        self.assertEqual(isolated, [])


class ProgressiveRunnerTests(unittest.TestCase):
    def test_dependent_rewrite_and_short_source_id_round_trip(self):
        """Controlled offline model, not a claim about real answer accuracy."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "inputs").mkdir()
            (root / "inputs/questions.json").write_text(json.dumps([
                {"id": "question_one", "question": "Where does the director live?"}]))
            corpus = [
                {"passage_id": "full_passage_A", "title": "Film",
                 "text": "Alice directed the film.", "sentences": ["Alice directed the film."]},
                {"passage_id": "full_passage_B", "title": "Alice",
                 "text": "Alice lives in Paris.", "sentences": ["Alice lives in Paris."]}]
            def node(name):
                return {"name": name, "type": "Thing", "description": name}
            def rel(a, b):
                return {"head": a, "tail": b, "type": "related",
                        "description": a + " relates to " + b, "strength": 8}
            extracted = {
                "full_passage_A": {"entities": {s: node(s) for s in ("film", "alice")},
                                   "relations": [rel("film", "alice")]},
                "full_passage_B": {"entities": {s: node(s) for s in ("alice", "paris")},
                                   "relations": [rel("alice", "paris")]}}
            graph = assemble(corpus, extracted)
            paths = list(iter_paths(graph))
            with (root / "path_repository.jsonl").open("w") as stream:
                for path in paths:
                    stream.write(json.dumps({**path, "questions": ["Who?", "Where?"]}) + "\n")
            vector = np.zeros(1024, dtype=np.float32)
            vector[0] = 1
            def index(items):
                return VectorIndex(items, np.stack([vector] * len(items)))
            runner = Runner.__new__(Runner)
            runner.root, runner.workers, runner.graph = root, 1, graph
            runner.indexes = {
                "hyperedges": index([{"id": p["passage_id"]} for p in corpus]),
                "entities": index([{"id": e} for e in graph["retained_entity_ids"]]),
                "paths": index([{"id": "question_vector", "path_id": paths[0]["id"]}])}
            class FakeEmbeddings:
                def encode(self, texts, **kwargs):
                    return np.stack([vector for _ in texts])
            runner.embed = FakeEmbeddings()
            observed = []
            class FakeClient:
                def call(self, *, stage, prompt, **kwargs):
                    payload = json.loads(prompt)
                    observed.append((stage, payload))
                    if stage == "direct":
                        value = {"sufficient": False, "answer": "", "supporting_passage_ids": []}
                    elif stage == "decomposition":
                        value = {"subquestions": [
                            {"question": "Who directed the film?", "entities": ["film"], "needs_external": False},
                            {"question": "Where does that director live?", "entities": [], "needs_external": True}]}
                    elif stage == "rewrite":
                        self_outer.assertEqual(payload["history"][0]["answer"], "Alice")
                        value = {"question": "Where does Alice live?", "entities": ["Alice"],
                                 "needs_external": False}
                    else:
                        answer = ("Alice" if stage == "subquestion"
                                  and payload["subquestion"] == "Who directed the film?" else "Paris")
                        self_outer.assertTrue(all(p["id"].startswith("s") for p in payload["passages"]))
                        value = {"answer": answer,
                                 "supporting_passage_ids": [payload["passages"][0]["id"]]}
                    return {"text": json.dumps(value), "request_sha256": "mock",
                            "stage": stage, "elapsed_s": 0, "cache_hit": False, "usage": {}}
            self_outer = self
            runner.client = FakeClient()
            runner.predict()
            output = json.loads((root / "predictions/question_one.json").read_text())
            self.assertEqual(output["route"], "progressive")
            self.assertEqual(output["answer"], "Paris")
            self.assertIn(output["supporting_passage_ids"][0],
                          {"full_passage_A", "full_passage_B"})
            self.assertEqual([s for s, _ in observed],
                             ["direct", "decomposition", "subquestion",
                              "rewrite", "subquestion", "final"])


if __name__ == "__main__":
    unittest.main()
