import ast
from collections import Counter
import json
from pathlib import Path
import re
import string
import tempfile
import unittest
from unittest.mock import patch

from qa import benchmark_evaluate as scoring


class NativeMetricAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.benchmark = json.loads((scoring.ROOT / "reproduction/benchmark.json").read_text(encoding="utf-8"))
        self.gold = self.folder / "gold.jsonl"
        rows = [
            {"_id": "q0", "answer": "The United Kingdom", "answer_id": "Qdemo", "type": "bridge"},
            {"_id": "q1", "answer": "missing", "answer_id": "", "type": "bridge"},
            {"_id": "q2", "answer": "yes", "answer_id": "", "type": "comparison"},
        ]
        self.gold.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def run_scores(self, answers):
        prediction = self.folder / "predictions.json"
        prediction.write_text(json.dumps({"answer": answers}), encoding="utf-8")
        with patch.object(scoring, "alias_mapping", return_value={"Qdemo": ["UK"]}):
            return scoring.evaluate(prediction, self.gold, ["q0", "q1", "q2"])

    def test_missing_answers_keep_denominator_and_binary_answer_rules(self):
        answer = "UK" if self.benchmark["id"] == "2wiki" else "United Kingdom"
        result = self.run_scores({"q0": answer, "q2": "yes it is"})
        self.assertEqual(result["denominator"], 3)
        self.assertEqual(result["missing_prediction_ids"], ["q1"])
        self.assertAlmostEqual(result["answer_em"], 1 / 3)
        self.assertAlmostEqual(result["answer_f1"], 1 / 3)
        self.assertEqual([row["id"] for row in result["per_question"]], ["q0", "q1", "q2"])
        self.assertEqual(result["per_question"][1]["answer_f1"], 0)
        if self.benchmark["id"] == "2wiki":
            self.assertEqual(result["per_question"][0]["canonical_answer_em"], 0)
            self.assertEqual(result["per_question"][0]["answer_em"], 1)

    def test_extra_and_nonstring_predictions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            self.run_scores({"q0": "answer", "extra": "answer"})
        with self.assertRaisesRegex(ValueError, "Non-string"):
            self.run_scores({"q0": {"answer": "unsupported"}})

    def test_functions_match_independent_original_native_update(self):
        dataset = self.benchmark["id"]
        source = scoring.ROOT / f"reproduction/provenance/official_evaluators/{dataset}_official_evaluate.py"
        wanted = {"normalize_answer", "f1_score", "exact_match_score", "eval_answer", "update_answer"}
        nodes = [node for node in ast.parse(source.read_text(encoding="utf-8")).body
                 if isinstance(node, ast.FunctionDef) and node.name in wanted]
        namespace = {"re": re, "string": string, "Counter": Counter}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
        exact, f1 = scoring.native_answer_functions(dataset)
        pairs = [
            ("The blue river", "blue river"), ("blue river valley", "blue river"),
            ("yes it is", "yes"), ("no", "yes"), ("", "answer"),
            ("O'Connor", "oconnor"), ("Paris, France", "Paris"),
        ]
        for prediction, label in pairs:
            metrics = {"em": 0.0, "f1": 0.0, "prec": 0.0, "recall": 0.0}
            namespace["update_answer"](metrics, prediction, [label] if dataset == "2wiki" else label)
            self.assertEqual(float(exact(prediction, label)), metrics["em"])
            self.assertAlmostEqual(f1(prediction, label)[0], metrics["f1"], places=14)


if __name__ == "__main__":
    unittest.main()
