"""Attribute actual usage and original prompt overhead without any model calls."""

from collections import Counter, defaultdict
import hashlib
import json

from reproduction.campaign import DEFAULT_RUN, read_records
from reproduction.runtime import utc_now, write_json
from reproduction.freeze_benchmark import DESTINATION


def main():
    records = [
        row for row in read_records(DEFAULT_RUN)
        if row.get("api_request_started") and row["status"] in ("completed", "failed", "refused")
    ]
    by_stage = {}
    for stage in sorted({row["stage"] for row in records}):
        subset = [row for row in records if row["stage"] == stage]
        reporting_calls = sum(row.get("usage") is not None for row in subset)
        totals = Counter()
        payloads = Counter(row["request_sha256"] for row in subset)
        for row in subset:
            usage = row.get("usage") or {}
            for key in ("input_tokens", "output_tokens", "total_tokens"):
                if isinstance(usage.get(key), int):
                    totals[key] += usage[key]
            totals["cached_input_tokens"] += (
                usage.get("input_tokens_details") or {}
            ).get("cached_tokens", 0)
        by_stage[stage] = {
            "finished_calls": len(subset),
            "calls_reporting_usage": reporting_calls,
            "calls_without_usage": len(subset) - reporting_calls,
            "reported_tokens": dict(totals),
            "mean_input_tokens_per_usage_reporting_call": (
                totals["input_tokens"] / reporting_calls if reporting_calls else None
            ),
            "mean_output_tokens_per_usage_reporting_call": (
                totals["output_tokens"] / reporting_calls if reporting_calls else None
            ),
            "cached_fraction_of_reported_input": (
                totals["cached_input_tokens"] / totals["input_tokens"] if totals["input_tokens"] else None
            ),
            "distinct_exact_request_payloads": len(payloads),
            "repeated_exact_payload_transmissions": sum(value - 1 for value in payloads.values()),
        }
    extraction = sorted(
        [row for row in records if row["stage"] == "graph_extraction"],
        key=lambda row: row["sent_unix"],
    )[:64]
    role_characters = Counter()
    prefixes = Counter()
    audited_ids = []
    for record in extraction:
        payload = json.loads(
            (DEFAULT_RUN / "requests" / (record["id"] + ".input.json")).read_text(encoding="utf-8")
        )
        for message in payload["messages"]:
            role_characters[message["role"]] += len(message["content"])
            if message["role"] == "system":
                prefixes[hashlib.sha256(message["content"].encode("utf-8")).hexdigest()] += 1
        audited_ids.append(record["id"])
    manifest = json.loads((DESTINATION / "manifest.json").read_text(encoding="utf-8"))
    report = {
        "generated_at": utc_now(),
        "no_new_model_calls": True,
        "complete": json.loads((DEFAULT_RUN / "state.json").read_text(encoding="utf-8")).get(
            "full_reproduction_completed", False
        ),
        "by_stage": by_stage,
        "original_extraction_prompt_profile": {
            "audited_call_ids": audited_ids,
            "audited_calls": len(extraction),
            "message_characters_by_role": dict(role_characters),
            "system_prompt_sha256_counts": dict(prefixes),
            "system_fraction_of_message_characters": (
                role_characters["system"] / sum(role_characters.values())
                if role_characters else None
            ),
            "note": "Character fractions are not token estimates or monetary savings",
        },
        "sample_caching": {
            "candidate_occurrences": manifest["candidate_occurrences"],
            "exact_unique_passages": manifest["unique_candidate_passages"],
            "repeated_occurrences": manifest["candidate_occurrences"] - manifest["unique_candidate_passages"],
            "note": "Original extraction cache already reuses exact passage text; retained, not a new scientific optimization",
        },
        "monetary_spend": None,
        "money_note": "Actual usage is recorded; no verified task-specific tariff available",
        "interpretation": [
            "Construction and inference are attributed separately; a larger QA sample can reuse the same graph only when its required candidate scope is covered",
            "64-way scheduling bounds concurrent calls; it does not reduce prompt tokens",
            "Original long system instructions repeat per chunk; provider-reported cached input is a subset of total input",
            "Packing multiple original requests into one prompt changes the extraction setting and requires a separately labeled quality evaluation",
        ],
    }
    write_json(DEFAULT_RUN / "cost_architecture_audit.json", report)
    print(json.dumps({
        "finished_calls": len(records), "by_stage": by_stage,
        "sampled_system_character_fraction": report["original_extraction_prompt_profile"][
            "system_fraction_of_message_characters"
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
