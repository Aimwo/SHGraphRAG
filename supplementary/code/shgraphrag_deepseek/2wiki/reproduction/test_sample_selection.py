import random
import unittest

from processor.sample_selection import sample_indices


class SampleSelectionTests(unittest.TestCase):
    def test_stable_ids_across_input_order_and_global_rng(self):
        ids = [f"q{i:04d}" for i in range(1000)]
        expected = random.Random(42).sample(sorted(ids), 500)
        self.assertEqual([ids[i] for i in sample_indices(ids)], expected)
        reordered = list(reversed(ids))
        previous_state = random.getstate()
        try:
            random.seed(987654)
            random.random()
            self.assertEqual([reordered[i] for i in sample_indices(reordered)], expected)
        finally:
            random.setstate(previous_state)
        self.assertNotEqual(expected, ids[:500])
        self.assertEqual(len(set(expected)), 500)

    def test_rejects_ambiguous_ids(self):
        for ids in (["a", "a"], ["", "a"], [None, "a"]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                sample_indices(ids, sample_size=1)

    def test_rejects_invalid_size(self):
        for size in (0, -1, 3):
            with self.subTest(size=size), self.assertRaises(ValueError):
                sample_indices(["a", "b"], sample_size=size)


if __name__ == "__main__":
    unittest.main()
