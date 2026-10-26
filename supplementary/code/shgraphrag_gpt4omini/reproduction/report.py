"""Summarize only recorded evidence, including incomplete runs and cache reuse."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from .client import write_json
from .data import read_jsonl


def optional_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def summarize(root: Path) -> dict:
    config = optional_json(root / "config.json")
    data = optional_json(root / "data_manifest.json")
    graph = optional_json(root / "graph_stats.json")
    repository = optional_json(root / "path_repository_manifest.json")
    metrics = optional_json(root / "evaluation/metrics.json")
    metrics.pop("per_sample", None)
    calls = read_jsonl(root / "api_calls.jsonl") if (root / "api_calls.jsonl").exists() else []
    cache_hits = (read_jsonl(root / "cache_hits.jsonl")
                  if (root / "cache_hits.jsonl").exists() else [])
    stages = defaultdict(Counter)
    for call in calls:
        stage = stages[call["stage"]]
        stage["attempts"] += 1
        stage[call["status"]] += 1
        for field in ("input_tokens", "output_tokens", "total_tokens"):
            stage[field] += call.get("usage", {}).get(field, 0)
        if call.get("usage", {}).get("cost") is not None:
            stage["reported_cost_usd"] += call["usage"]["cost"]
            stage["attempts_with_reported_cost"] += 1
        stage["reasoning_tokens"] += (call.get("usage", {})
                                     .get("output_tokens_details", {}).get("reasoning_tokens", 0))
        stage["request_seconds_sum"] += call["elapsed_s"]
    predictions = [optional_json(p) for p in (root / "predictions").glob("*.json")]
    times = [p["elapsed_s"] for p in predictions if p.get("status") == "completed"]
    def percentiles(values):
        return ({name: float(np.percentile(values, pct))
                 for name, pct in (("p50", 50), ("p90", 90), ("p95", 95), ("p99", 99))}
                | {"mean": float(np.mean(values)), "max": max(values)}
                if values else {})
    components = {field: percentiles([
        p.get("component_seconds", {}).get(field, 0) for p in predictions])
        for field in ("embedding_s_sum", "vector_search_s_sum", "llm_stage_s_sum")}
    storage = {}
    for label, paths in {
        "graph": [root / "graph.json"],
        "indexes": list((root / "indexes").glob("*")),
        "path_repository": [root / "path_repository.jsonl"],
        "extractions": list((root / "extractions").glob("*")),
        "path_batches": list((root / "path_batches").glob("*")),
        "predictions": list((root / "predictions").glob("*")),
    }.items():
        storage[label] = sum(p.stat().st_size for p in paths if p.is_file())
    timings = read_jsonl(root / "timing.jsonl") if (root / "timing.jsonl").exists() else []
    progress = read_jsonl(root / "progress.jsonl") if (root / "progress.jsonl").exists() else []
    phase_starts = {}
    phase_attempts = []
    for event in progress:
        stage = event["stage"]
        if event.get("status") == "running":
            phase_starts[stage] = event["time_unix"]
        elif event.get("status") in ("completed", "failed") and stage in phase_starts:
            phase_attempts.append({
                "stage": stage, "status": event["status"],
                "elapsed_s": event["time_unix"] - phase_starts.pop(stage)})
    result = {
        "run": str(root.resolve()), "config": config, "data": data,
        "graph": graph, "path_repository": repository, "metrics": metrics,
        "coverage": {"extractions": len(list((root / "extractions").glob("*.json"))),
                     "predictions": len(times), "expected_questions": data.get("questions"),
                     "expected_passages": data.get("passages")},
        "api_stages": {name: dict(value) for name, value in stages.items()},
        "api_concurrency_observed_max": max(
            (c.get("active_http_at_dispatch", 0) for c in calls), default=0),
        "api_attempt_latency_s": percentiles([c["elapsed_s"] for c in calls]),
        "reported_cost_usd": sum(c.get("usage", {}).get("cost") or 0 for c in calls),
        "cost_scope": "Reported usage only; unreported failed/interrupted remote usage is unknown.",
        "upstream_provider_attempts": dict(Counter(
            c.get("upstream_provider") for c in calls)),
        "api_cache_hits": len(cache_hits), "wall_time_records": timings,
        "runtime_environment": optional_json(root / "runtime_environment.json"),
        "resource_summary": optional_json(root / "resource_summary.json"),
        "query_component_latency_s": components,
        "query_route_counts": dict(Counter(p.get("route") for p in predictions)),
        "query_latency_by_route_s": {
            route: percentiles([p["elapsed_s"] for p in predictions
                                if p.get("route") == route])
            for route in sorted({p.get("route", "") for p in predictions})},
        "phase_throughput": {
            label: {"items": count, "wall_s": sum(
                t["elapsed_s"] for t in timings if t["stage"] == stage)}
            for label, count, stage in (
                ("extraction", data.get("passages", 0), "extraction"),
                ("path_questions", repository.get("questions", 0), "path_questions"),
                ("prediction", len(times), "prediction"))},
        "phase_attempts_including_failures": phase_attempts,
        "construction_wall_s_including_failed_attempts": sum(
            item["elapsed_s"] for item in phase_attempts
            if item["stage"] in ("extraction", "graph", "path_questions", "index")),
        "latency_s": percentiles(times),
        "storage_bytes": storage,
        "complete": bool(metrics and metrics["missing_or_failed"] == 0
                         and metrics["completed"] == data.get("questions")
                         and len(list((root / "extractions").glob("*.json"))) == data.get("passages")
                         and repository
                         and repository["paths_completed"] == repository["paths_expected"]),
        "limitations": [
            "OpenRouter GPT-4o-mini model substitution; not Gemini 2.0 Flash numeric reproduction.",
            "New seed-20261020 samples; not the undocumented original paper sample.",
            "Candidate passage union only; no full open-domain corpus evaluation.",
            "Isolated JSON graph, SQLite cache and exact cosine implementation; not Neo4j latency.",
            "At most three original source hyperedges accompany the top-three direct paths.",
            "Any cache reuse is reported; wall timings are not pure cold-build measurements.",
            "Passage support diagnostics are not official sentence-level support metrics.",
        ],
    }
    if config.get("method_version") == "source_alias_merge_global_graph_v2":
        result["entity_resolution"] = optional_json(root / "entity_resolution/manifest.json")
        result["limitations"] += [
            "Unlabelled candidate groups define alias-candidate construction scope only; "
            "retrieval uses the global union without question membership filters.",
            "Positive local alias decisions are reconciled across overlapping groups, "
            "with corpus-frequency canonical names rather than question-suffixed nodes.",
            "Entity vectors use canonical names as in the supplied source; all source "
            "descriptions remain available. No separate entity-summary stage exists "
            "in the supplied active build/index path.",
            "Independent subquestions execute in parallel waves under the same shared request limit.",
        ]
    else:
        result["limitations"] += [
            "Exact normalized-name identity; no LLM alias/entity merging.",
            "Entity index uses first source description; all descriptions remain in the graph.",
            "Context-independent subquestions execute in order; questions run concurrently.",
            "Historical adaptation pilot: not accepted as the final method implementation.",
        ]
    write_json(root / "summary.json", result)
    lines = [
        f"# SHGraphRAG GPT-4o-mini reproduction: {config.get('dataset', root.name)}",
        "", f"Run: `{root.resolve()}`",
        "", f"Status: {'complete at this scope' if result['complete'] else 'incomplete'}",
        "", f"Model: `{config.get('model')}`; reasoning effort: `{config.get('reasoning_effort')}`.",
        f"Scope: {data.get('questions', 0)} questions, {data.get('passages', 0)} candidate passages; "
        "global union retrieval.",
        "", "| Recorded item | Value |", "|---|---:|",
        f"| Extraction coverage | {result['coverage']['extractions']} / {data.get('passages', 0)} |",
        f"| Prediction coverage | {len(times)} / {data.get('questions', 0)} |",
        f"| Full / sparse entities | {graph.get('full_entities', '-')} / {graph.get('sparse_entities', '-')} |",
        f"| Full / sparse triples | {graph.get('full_triples', '-')} / {graph.get('sparse_triples', '-')} |",
        f"| Two-edge paths | {graph.get('two_edge_paths', '-')} |",
        f"| Generated path questions | {repository.get('questions', '-')} |",
        f"| Maximum observed concurrent API requests | {result['api_concurrency_observed_max']} |",
        f"| Reported API cost (USD) | {result['reported_cost_usd']:.6f} |",
    ]
    if metrics:
        lines += [
            f"| Answer EM (Hotpot protocol) | {metrics['primary_em']:.4f}% |",
            f"| Answer F1 (Hotpot protocol) | {metrics['primary_f1']:.4f}% |",
            f"| Alias-aware EM / F1 | {metrics['native_alias_em']:.4f}% / {metrics['native_alias_f1']:.4f}% |",
            f"| Direct answers | {metrics['direct_count']} |",
        ]
    if times:
        lines += [f"| Question latency mean / P95 | {result['latency_s']['mean']:.3f}s / "
                  f"{result['latency_s']['p95']:.3f}s |"]
    lines += ["", "## API usage", "",
              "| Stage | Attempts | Completed | Failed | Input tokens | Output tokens | Reasoning tokens |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name, row in sorted(stages.items()):
        lines.append(f"| {name} | {row['attempts']} | {row['completed']} | {row['failed']} | "
                     f"{row['input_tokens']} | {row['output_tokens']} | {row['reasoning_tokens']} |")
    lines += ["", f"API cache hits: {len(cache_hits)}.", "",
              "Construction wall time, including recorded failed attempts and resumptions: "
              f"{result['construction_wall_s_including_failed_attempts']:.3f} seconds.",
              "This includes observed cache reuse and is not a pure cold-build benchmark.", "",
              "## Interpretation limits", ""]
    lines += ["- " + item for item in result["limitations"]]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    result = summarize(args.run)
    print(json.dumps({k: result[k] for k in ("coverage", "graph", "metrics", "complete")},
                     ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
