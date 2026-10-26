"""Offline acceptance checks for the added native datasets and evaluators."""
import importlib.util
import unittest

from .common import PROJECT, ROOT, read_json, read_jsonl
from .evaluate import answer_scores, score_records
from .prepare import validate_inputs


class NativeDatasetTests(unittest.TestCase):
    def test_native_metric_primitives_match_both_supplied_reference_scripts(self):
        qa = ROOT.parents[2] / "code/shgraphrag_gpt4omini/qa"
        pairs = [
            ("The Paris.", "paris"), ("yes indeed", "yes"), ("yes", "yes"),
            ("", "some answer"), ("first second", "second third"), ("", ""),
            ("no", "noanswer"), ("a", ""), ("2001-02", "200102"),
        ]
        for dataset, filename in (
                ("hotpotqa", "hotpot_evaluate_v1.py"),
                ("2wiki", "2wikimultihop_evaluate.py")):
            spec = importlib.util.spec_from_file_location("reference_" + dataset, qa / filename)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            for prediction, gold in pairs:
                with self.subTest(dataset=dataset, prediction=prediction, gold=gold):
                    em, f1 = answer_scores(prediction, gold, dataset)
                    self.assertEqual(em, float(module.exact_match_score(prediction, gold)))
                    self.assertAlmostEqual(f1, module.f1_score(prediction, gold)[0])

    def test_complete_frozen_inputs_reconstruct_from_official_sources(self):
        validated = validate_inputs()
        self.assertTrue(validated["passed"])
        self.assertEqual(validated["questions"], 1000)
        self.assertEqual(validated["seed"], 42)
        queries = read_jsonl(ROOT / "inputs/queries.jsonl")
        self.assertTrue(all(set(query) == {"id", "question"} for query in queries))
        self.assertEqual([query["id"] for query in queries], read_json(ROOT / "inputs/ids.json"))

    def test_missing_failed_and_extra_ids_preserve_denominator(self):
        gold = [
            {"_id": "one", "answer": "Paris", "type": "bridge",
             "context": [["Paris", ["Paris is in France."]]], "supporting_facts": [["Paris", 0]]},
            {"_id": "two", "answer": "London", "type": "bridge",
             "context": [["London", ["London is in England."]]], "supporting_facts": [["London", 0]]},
        ]
        predictions = {"one": {"status": "completed", "answer": "Paris"}}
        metrics = score_records(["one", "two"], gold, predictions, dataset="hotpotqa")
        self.assertEqual(metrics["denominator"], 2)
        self.assertEqual(metrics["single_reference_em"], 50)
        self.assertEqual(metrics["missing_or_failed"], 1)
        predictions["two"] = {"status": "failed", "answer": "London"}
        metrics = score_records(["one", "two"], gold, predictions, dataset="hotpotqa")
        self.assertEqual(metrics["single_reference_em"], 50)
        predictions["extra"] = {"status": "completed", "answer": "London"}
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            score_records(["one", "two"], gold, predictions, dataset="hotpotqa")

    def test_two_wiki_aliases_are_evaluation_only_and_separate_from_primary(self):
        gold = [{"_id": "one", "answer": "United Kingdom", "answer_id": "Q145",
                 "type": "compositional", "context": [["UK", ["A country."]]],
                 "supporting_facts": [["UK", 0]]}]
        predictions = {"one": {"status": "completed", "answer": "Britain"}}
        metrics = score_records(["one"], gold, predictions, dataset="2wiki",
                                aliases={"Q145": {"aliases": ["Britain"], "demonyms": ["British"]}})
        self.assertEqual(metrics["single_reference_em"], 0)
        self.assertEqual(metrics["native_alias_em"], 100)
        self.assertEqual(metrics["native_alias_f1"], 100)

    def test_all_native_gold_answers_score_exactly_and_keep_draw_order(self):
        # Synthetic answers remain in memory; never populate a formal run.
        manifest = read_json(ROOT / "inputs/manifest.json")
        ids, gold = read_json(ROOT / "inputs/ids.json"), read_jsonl(ROOT / "inputs/gold.jsonl")
        predictions = {row["_id"]: {"status": "completed", "answer": row["answer"]} for row in reversed(gold)}
        metrics = score_records(ids, gold, predictions, dataset=manifest["dataset_id"])
        self.assertEqual(metrics["denominator"], 1000)
        self.assertEqual(metrics["single_reference_em"], 100)
        self.assertEqual(metrics["single_reference_f1"], 100)
        self.assertEqual([row["id"] for row in metrics["per_sample"]], ids)


if __name__ == "__main__":
    unittest.main()
