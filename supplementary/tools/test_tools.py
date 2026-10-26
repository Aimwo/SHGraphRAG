"""Standard-library tests for publication orchestration and API boundaries."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from reproduce import command_plan


class PlanTests(unittest.TestCase):
    def test_hyper_full_campaign_keeps_one_frozen_dataset_selection(self):
        plan = command_plan("hyper-gpt4omini", "all", Path("prepared"),
                            "check_01", "run", sys.executable)
        command = plan["commands"][0]
        self.assertEqual(command[command.index("--datasets") + 1:],
                         ["hotpotqa", "2wiki", "musique"])
        self.assertEqual(plan["environment_overrides"]["OPENROUTER_MODEL"], "openai/gpt-4o-mini")
        self.assertNotIn("OPENROUTER_API_KEY", plan["environment_overrides"])
        self.assertFalse(plan["model_calls_allowed"])

    def test_full_development_source_not_resampled_handoff(self):
        plan = command_plan("sh-deepseek", "2wiki", Path("prepared"),
                            "check_01", "prepare", sys.executable)
        self.assertTrue(plan["environment_overrides"]["SHGRAPH_FULL_DEV_PARQUET"].endswith("2wiki.parquet"))
        self.assertNotIn("questions.parquet",
                         plan["environment_overrides"]["SHGRAPH_FULL_DEV_PARQUET"])

    def test_evaluation_executes_scorer_without_live_pipeline(self):
        plan = command_plan("sh-deepseek", "musique", Path("prepared"),
                            "check_01", "evaluate", sys.executable)
        self.assertIn("m.evaluate(", plan["commands"][0][4])
        self.assertNotIn("reproduction.campaign", " ".join(plan["commands"][0]))

    def test_execute_run_without_opt_in_stops_before_child(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, str(Path(__file__).with_name("reproduce.py")),
                 "--profile", "sh-gpt4omini", "--dataset", "hotpotqa",
                 "--data-root", directory, "--phase", "run", "--execute"],
                capture_output=True, text=True, encoding="utf-8")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no child was started", result.stderr)


if __name__ == "__main__":
    unittest.main()
