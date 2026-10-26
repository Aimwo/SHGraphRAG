"""Check real DatasetProcessor ID/chunk alignment, without optional model loads.

The actual local Parquet loader, sampler, process_data and process_dataset
methods execute. The unused HanLP tokenizer now loads lazily in production.
This is a preprocessing check, not a graph/LLM end-to-end test.
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Confine all Hugging Face caches and disable remote access before importing it.
os.environ["HF_HOME"] = str(ROOT / ".cache/huggingface")
os.environ["HF_DATASETS_CACHE"] = str(ROOT / ".cache/huggingface/datasets")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import pyarrow.parquet as pq

from config.settings import DATASET_DIR
from reproduction.freeze_musique_sample import DESTINATION, require, write_json


def main() -> None:
    from processor.dataset_processor import DatasetProcessor
    processor = DatasetProcessor(str(DATASET_DIR))
    processed = processor.process_dataset()
    ids = json.loads((DESTINATION / "ids.json").read_text(encoding="utf-8"))
    rows = pq.read_table(DESTINATION / "questions.parquet").to_pylist()
    require(list(processor.dataset["id"]) == ids, "Processor sampled different IDs/order")
    require([row["question_id"] for row in processed] == ids, "Processed IDs/order differ")
    require(len(processed) == len(rows) == 1000, "Processor did not process 1000 questions")
    for result, row in zip(processed, rows, strict=True):
        expected_chunks = [
            title + "\n" + "\n".join(sentences)
            for title, sentences in zip(
                row["context"]["title"], row["context"]["sentences"], strict=True
            )
        ]
        require(list(result["chunks"]) == expected_chunks, "Candidate chunk text/order changed")
        require(result["chunk_count"] == len(expected_chunks), "Candidate count changed")
        require("chunk_error" not in result, "Upstream processing raised a chunk error")
    report = {
        "passed": True,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "source": str(DATASET_DIR),
        "processed_questions": len(processed),
        "candidate_occurrences": sum(len(row["chunks"]) for row in processed),
        "ids_match_frozen_order": True,
        "chunks_match_upstream_text_format": True,
        "real_code_executed": [
            "datasets.load_dataset(local Parquet)",
            "DatasetProcessor.__init__ and sample_indices",
            "DatasetProcessor.process_data",
            "DatasetProcessor.process_dataset",
        ],
        "stubbed": [],
        "hanlp_not_loaded": processor._chunker is None,
        "llm_calls": 0,
        "is_end_to_end_reproduction": False,
    }
    write_json(ROOT / "reproduction/audit/processor_validation.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
