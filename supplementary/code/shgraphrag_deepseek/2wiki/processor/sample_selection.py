"""Deterministic ID sampling, independent of source row order and global RNG."""

import random
from collections.abc import Sequence


def sample_indices(
    question_ids: Sequence[str], sample_size: int = 500, seed: int = 42
) -> list[int]:
    """Sample sorted unique IDs, retaining the RNG draw order in the result.

    This is the only scientific preprocessing change requested for the fresh
    upstream reproduction. It never filters or edits a question's passages.
    """
    ids = list(question_ids)
    if any(not isinstance(qid, str) or not qid for qid in ids):
        raise ValueError("Question IDs must be nonempty strings")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate question IDs: sample identity is ambiguous")
    if not 0 < sample_size <= len(ids):
        raise ValueError("Sample size must be positive and no larger than the source")
    row_by_id = {qid: index for index, qid in enumerate(ids)}
    sampled_ids = random.Random(seed).sample(sorted(ids), sample_size)
    return [row_by_id[qid] for qid in sampled_ids]
