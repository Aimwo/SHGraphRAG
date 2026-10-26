"""Meaningful evaluator checks against native gold, not benchmark tuning."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from revision_openrouter.evaluate import evaluate_run, scores
from revision_openrouter.run import LOCK, PROJECT, read_json, read_jsonl
from revision_openrouter.transport import write_json


class EvaluationTests(unittest.TestCase):
    def test_exact_denominator_all_three_native_formats(self):
        for entry in read_json(LOCK)["datasets"]:
            with self.subTest(dataset=entry["dataset"]), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = PROJECT / entry["sources"]["native_gold_evaluation_only"]["path"]
                gold = read_jsonl(source) if entry["dataset"] == "musique" else read_json(source)
                ids = [row.get("_id") or row["id"] for row in gold[:2]]
                write_json(root / "data_manifest.json", {"ids": ids})
                write_json(root / "predictions/first.json", {
                    "id": ids[0], "status": "completed", "answer": gold[0]["answer"],
                    "retrieved_passage_ids": []})
                result = evaluate_run(root, PROJECT, entry)
                self.assertEqual(result["expected"], 2)
                self.assertEqual(result["completed"], 1)
                self.assertEqual(result["missing_or_failed"], 1)
                self.assertEqual(result["primary_em"], 50)
                self.assertEqual(result["primary_f1"], 50)
                self.assertEqual(result["native_alias_em"], 50)
                # An otherwise plausible extra prediction cannot expand or
                # silently change the evaluation subset.
                write_json(root / "predictions/extra.json", {
                    "id": "not_in_frozen_ids", "status": "completed", "answer": ""})
                with self.assertRaises(RuntimeError):
                    evaluate_run(root, PROJECT, entry)

    def test_primary_parity_with_official_metric_primitives(self):
        qa = Path(__file__).resolve().parents[3] / "code/shgraphrag_gpt4omini/qa"
        spec = importlib.util.spec_from_file_location("official_hotpot", qa / "hotpot_evaluate_v1.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        examples = [("The Paris.", "paris"), ("yes indeed", "yes"), ("yes", "yes"),
                    ("", "some answer"), ("first second", "second third"),
                    ("no", "noanswer"), ("a", ""), ("2001-02", "200102")]
        for prediction, gold in examples:
            em, f1 = scores(prediction, gold)
            self.assertEqual(em, module.exact_match_score(prediction, gold))
            self.assertAlmostEqual(f1, module.f1_score(prediction, gold)[0])


if __name__ == "__main__":
    unittest.main()
