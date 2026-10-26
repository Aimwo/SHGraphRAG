import unittest

from .efficiency import ledger
from .telemetry import process_resources


class EfficiencyTests(unittest.TestCase):
    def test_failed_usage_and_cache_hits_are_kept_separate(self):
        result = ledger([
            {"status": "completed", "elapsed_s": 2,
             "usage": {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110},
             "usage_raw": {"cost": 0.1}},
            {"status": "failed", "elapsed_s": 3,
             "usage": {"input_tokens": 40, "output_tokens": 5, "total_tokens": 45}},
            {"status": "failed", "elapsed_s": 4},
        ], [{"request_sha256": "reused"}])
        self.assertEqual(result["attempts"], 3)
        self.assertEqual(result["successful_calls"], 1)
        self.assertEqual(result["failed_attempts"], 2)
        self.assertEqual(result["input_tokens"], 140)
        self.assertEqual(result["total_tokens"], 155)
        self.assertEqual(result["attempts_without_reported_usage"], 1)
        self.assertEqual(result["local_response_cache_hits"], 1)
        self.assertEqual(result["successful_http_latency"]["mean_s"], 2)
        self.assertAlmostEqual(result["provider_reported_cost_sum"], 0.1)

    def test_native_resource_probe_returns_plausible_values(self):
        sample = process_resources()
        self.assertGreater(sample["pid"], 0)
        self.assertGreaterEqual(sample["process_cpu_s"], 0)
        if "resident_bytes" in sample:
            self.assertGreater(sample["resident_bytes"], 1024 ** 2)
            self.assertGreaterEqual(sample["peak_resident_bytes"], sample["resident_bytes"])


if __name__ == "__main__":
    unittest.main()
