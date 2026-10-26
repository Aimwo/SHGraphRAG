"""Complete the approved official-build + supplied-QA MuSiQue reproduction."""

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import platform
import re
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
import zipfile

import numpy as np
import psutil

from reproduction.database import (
    NEO4J_HOME, assert_owned_database, configure_process_environment,
)
from reproduction.freeze_benchmark import DESTINATION, validate
from reproduction.runtime import (
    ROOT, append_jsonl, call_context, run_root, utc_now, write_json,
)
from reproduction.relay_config import load_relay_config, config_identity, resume_config_variants
from config.settings import (
    EMBEDDING_MAX_IN_FLIGHT, SAMPLE_SIZE, SAMPLE_SEED, TEXT_MAX_WORKERS, TEXT_DISPATCH_WORKERS,
    TEXT_RECOVERABLE_ATTEMPTS,
)

DEFAULT_RUN = ROOT / "reproduction/runs/2wiki1000_seed42_deepseek-relay-14ac1cd4ea79_20261024"


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_files():
    """Capture only this exported implementation, independent of outer Git layout."""
    excluded = {"runs", "artifacts", "audit", "__pycache__", ".runtime", ".cache"}
    result = []
    for folder in ("build", "community", "config", "graph", "model", "processor", "qa", "reproduction"):
        for path in (ROOT / folder).rglob("*"):
            if (path.is_file() and path.suffix in (".py", ".txt", ".json")
                    and not excluded.intersection(path.relative_to(ROOT).parts)):
                result.append(path.relative_to(ROOT).as_posix())
    return sorted(result)


def snapshot_source(root):
    paths = source_files()
    hashes = {name: file_hash(ROOT / name) for name in paths}
    code_hash = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    destination = root / "source"
    destination.mkdir(exist_ok=True)
    write_json(destination / f"{code_hash}.hashes.json", hashes)
    archive_path = destination / f"{code_hash}.zip"
    if not archive_path.exists():
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in paths:
                # Only explicitly listed source; .env and runtime/credentials
                # cannot enter this archive through a recursive directory scan.
                archive.write(ROOT / name, name)
    return code_hash


def read_records(root):
    return [json.loads(path.read_text(encoding="utf-8")) for path in (root / "requests").glob("*.record.json")]


def usage_summary(root):
    from model.deepseek_relay import active_refusal_chunk_hashes
    records = read_records(root)
    physical = [row for row in records if row.get("api_request_started")]
    totals = Counter()
    field_coverage = Counter()
    per_stage = defaultdict(Counter)
    missing_usage = []
    events = []
    for row in physical:
        stage = row.get("stage", "unspecified")
        per_stage[stage]["physical_requests"] += 1
        per_stage[stage][row["status"]] += 1
        usage = row.get("usage")
        if usage is None:
            missing_usage.append(row["id"])
        else:
            for key in ("input_tokens", "output_tokens", "total_tokens"):
                if isinstance(usage.get(key), int):
                    totals[key] += usage[key]
                    per_stage[stage][key] += usage[key]
                    field_coverage[key] += 1
            if isinstance((usage.get("input_tokens_details") or {}).get("cached_tokens"), int):
                field_coverage["cached_input_tokens"] += 1
            if isinstance((usage.get("output_tokens_details") or {}).get("reasoning_tokens"), int):
                field_coverage["reasoning_output_tokens"] += 1
            totals["cached_input_tokens"] += (usage.get("input_tokens_details") or {}).get("cached_tokens", 0)
            totals["reasoning_output_tokens"] += (usage.get("output_tokens_details") or {}).get("reasoning_tokens", 0)
        began = row.get("sent_unix")
        ended = row.get("received_unix")
        if ended is None and row.get("finished_at"):
            ended = datetime.fromisoformat(row["finished_at"]).timestamp()
        if began is not None and ended is not None:
            events.extend([(began, 1), (ended, -1)])
    active = peak = 0
    for _, change in sorted(events):
        active += change
        peak = max(peak, active)
    return {
        "physical_requests": len(physical),
        "completed_requests": sum(row["status"] == "completed" for row in physical),
        "failed_requests": sum(row["status"] == "failed" for row in physical),
        "refused_requests": sum(row["status"] == "refused" for row in physical),
        "content_filter_physical_attempts": sum(row.get("finish_reason") == "content_filter" for row in physical),
        "active_content_filter_extraction_chunk_hashes": sorted(active_refusal_chunk_hashes(root)),
        "content_filter_extraction_chunk_hashes": sorted({
            row["chunk_hash"] for row in physical
            if row.get("finish_reason") == "content_filter"
            and row.get("stage") == "graph_extraction" and row.get("chunk_hash")
        }),
        "outstanding_request_records": sum(row["status"] in ("in_flight", "waiting_for_slot") for row in records),
        "reported_token_totals": dict(totals),
        "reported_usage_field_call_counts": dict(field_coverage),
        "optional_usage_fields_note": "A zero sum with zero reporting calls means unknown, not measured zero",
        "requests_missing_usage": len(missing_usage),
        "missing_usage_call_ids": missing_usage,
        "observed_completed_interval_peak_concurrency": peak,
        "global_request_gate_capacity": TEXT_MAX_WORKERS,
        "by_stage": {name: dict(counts) for name, counts in per_stage.items()},
        "monetary_cost": None,
        "monetary_cost_note": "No verified relay tariff supplied; tokens are actual returned usage",
    }


def graph_snapshot(graph):
    counts = graph.query("""
    MATCH (n)
    RETURN count(n) AS all_nodes,
      count(CASE WHEN n:__Document__ THEN 1 END) AS documents,
      count(CASE WHEN n:__Chunk__ THEN 1 END) AS chunks,
      count(CASE WHEN n:__Entity__ THEN 1 END) AS entities
    """)[0]
    counts.update(graph.query("MATCH ()-[r]->() RETURN count(r) AS all_relationships")[0])
    counts["entity_relationships"] = graph.query(
        "MATCH (:__Entity__)-[r]->(:__Entity__) RETURN count(r) AS n"
    )[0]["n"]
    counts["vector_indexes"] = graph.query(
        "SHOW INDEXES YIELD name, type, state WHERE type = 'VECTOR' RETURN name, type, state"
    )
    counts["path_question_repository"] = {
        "implemented_in_selected_source": False, "records": 0, "bytes": 0,
        "note": "Official construction step is #todo; supplied QA queries two-hop paths online",
    }
    counts["database_disk_bytes"] = sum(
        path.stat().st_size for path in (NEO4J_HOME / "data").rglob("*") if path.is_file()
    )
    return counts


def validate_graph_alignment(graph):
    from graph.core import generate_hash
    import pyarrow.parquet as pq
    rows = pq.read_table(DESTINATION / "questions.parquet").to_pylist()
    manifest = json.loads((DESTINATION / "manifest.json").read_text(encoding="utf-8"))
    source_name = Path(manifest["full_development_parquet"]["path"]).name
    expected = {}
    for row in rows:
        expected[source_name + "_" + row["id"]] = {
            generate_hash(title + "\n" + "\n".join(sentences))
            for title, sentences in zip(row["context"]["title"], row["context"]["sentences"], strict=True)
        }
    found = graph.query("""
    MATCH (d:__Document__)
    OPTIONAL MATCH (d)--(c:__Chunk__)
    RETURN d.fileName AS filename, collect(DISTINCT c.id) AS chunk_ids
    """)
    actual = {row["filename"]: set(row["chunk_ids"]) for row in found}
    if actual != expected:
        bad = [name for name in expected if actual.get(name) != expected[name]]
        raise ValueError(f"Graph document/candidate ID alignment failed: {bad[:5]}")
    return {
        "passed": True, "documents": len(actual),
        "distinct_question_passage_links": sum(len(values) for values in actual.values()),
    }


def monitor_resources(root, stopped):
    process = psutil.Process()
    java = psutil.Process(assert_owned_database()["java_pid"])
    while not stopped.wait(2):
        try:
            append_jsonl(root / "resources.jsonl", {
                "at": utc_now(), "stage": os.environ.get("SHGRAPH_STAGE"),
                "controller_rss_bytes": process.memory_info().rss,
                "neo4j_rss_bytes": java.memory_info().rss,
                "available_system_memory_bytes": psutil.virtual_memory().available,
                "controller_cpu_times": list(process.cpu_times()),
                "neo4j_cpu_times": list(java.cpu_times()),
            })
        except psutil.Error:
            return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--local-preflight-only", action="store_true")
    args = parser.parse_args()
    os.environ["SHGRAPH_RUN_ROOT"] = str(args.run_dir.resolve())
    relay_config = load_relay_config()
    configure_process_environment()
    root = run_root()
    owner = assert_owned_database()
    code_hash = snapshot_source(root)
    sample_manifest = json.loads((DESTINATION / "manifest.json").read_text(encoding="utf-8"))
    validate(
        Path(sample_manifest["source_parquet"]["path"]),
        Path(sample_manifest["native_gold_source"]["path"]), DESTINATION,
    )
    config = {
        "provider": relay_config["provider"], "model": relay_config["model"],
        "protocol": relay_config["protocol"], "base_url": relay_config["base_url"],
        "relay_config_sha256": config_identity(relay_config),
        "credential_id": relay_config["credential_id"],
        "generation_parameters": relay_config["generation_parameters"],
        "reasoning_effort": relay_config["generation_parameters"].get("reasoning_effort"),
        "temperature": None, "max_output_tokens": None,
        "unspecified_generation_parameters": "Omitted; public source supplies no values",
        "max_actual_in_flight_requests": TEXT_MAX_WORKERS,
        "sampling_seed": SAMPLE_SEED, "question_count": SAMPLE_SIZE,
        "sample_ids_sha256": file_hash(DESTINATION / "ids.json"),
        "sample_manifest_sha256": file_hash(DESTINATION / "manifest.json"),
        "upstream_commit": "3ed1685fefb1295c43c7fca0bb306e4d5fd08e77",
        "working_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
        "source_bundle_sha256": code_hash, "python": sys.version,
        "platform": platform.platform(), "processor": platform.processor(),
        "logical_cpu_count": psutil.cpu_count(), "physical_cpu_count": psutil.cpu_count(logical=False),
        "system_memory_bytes": psutil.virtual_memory().total,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "database_owner": owner,
        "source_combination": "Official GitHub build + supplied local QA/evaluators, with documented repairs",
        "local_qa_provenance": json.loads((ROOT / "reproduction/provenance/local_qa_sources.json").read_text(encoding="utf-8")),
    }
    from config import settings
    config["upstream_effective_settings"] = {
        name: getattr(settings, name) for name in (
            "CHUNK_SIZE", "OVERLAP", "MAX_WORKERS", "BATCH_SIZE",
            "ENTITY_BATCH_SIZE", "CHUNK_BATCH_SIZE", "EMBEDDING_BATCH_SIZE", "LLM_BATCH_SIZE",
            "TOP_K", "SIMILARITY_THRESHOLD", "GDS_MEMORY_LIMIT", "GDS_CONCURRENCY",
            "GDS_NODE_COUNT_LIMIT", "GDS_TIMEOUT_SECONDS",
        )
    }
    config["original_prompt_file_sha256"] = file_hash(ROOT / "config/prompt.py")
    config["engineering_scheduling"] = {
        "text_dispatch_workers": TEXT_DISPATCH_WORKERS,
        "recoverable_attempts_per_logical_call": TEXT_RECOVERABLE_ATTEMPTS,
        "recoverable_retry_delay_seconds": 0,
        "global_per_chunk_queue": True, "parallel_independent_document_merges": True,
        "local_embedding_max_in_flight": EMBEDDING_MAX_IN_FLIGHT,
        "local_embedding_max_items_per_request": 32,
        "local_embedding_runner_refusal_max_attempts": 3,
        "local_embedding_retry_delays_seconds": [1, 2],
        "resume_document_merge_checkpoints": True,
        "resume_exact_prior_merge_response_replay": False,
        "cross_question_prompt_concatenation": False,
        "tuple_codec": "original", "qa_prompts": "supplied original",
    }
    previous_config = root / "effective_config.json"
    if previous_config.exists():
        previous = json.loads(previous_config.read_text(encoding="utf-8"))
        for field in ("model", "credential_id", "sample_ids_sha256", "sample_manifest_sha256"):
            if previous[field] != config[field]:
                raise ValueError(f"Refusing to reuse a run with changed {field}")
        compatible_identities = {
            config_identity(variant): reason
            for variant, reason in resume_config_variants(relay_config)
        }
        if previous["relay_config_sha256"] not in compatible_identities:
            raise ValueError("Refusing to reuse a run with changed relay_config_sha256")
        if previous["relay_config_sha256"] != config["relay_config_sha256"]:
            append_jsonl(root / "relay_config_transition_events.jsonl", {
                "at": utc_now(),
                "previous_relay_config_sha256": previous["relay_config_sha256"],
                "current_relay_config_sha256": config["relay_config_sha256"],
                "reason": compatible_identities[previous["relay_config_sha256"]],
                "previous_request_capacity": previous["max_actual_in_flight_requests"],
                "current_request_capacity": TEXT_MAX_WORKERS,
            })
        append_jsonl(root / "config_revisions.jsonl", previous)
    write_json(root / "effective_config.json", config)
    state = {
        "status": "running", "stage": "local_preflight", "pid": os.getpid(),
        "process_create_time": psutil.Process().create_time(), "started_at": utc_now(),
        "source_bundle_sha256": code_hash, "full_reproduction_completed": False,
    }
    write_json(root / "state.json", state)
    stopped = threading.Event()
    monitor = threading.Thread(target=monitor_resources, args=(root, stopped), daemon=True)
    monitor.start()

    @contextmanager
    def stage(name):
        os.environ["SHGRAPH_STAGE"] = name
        state.update(stage=name, updated_at=utc_now())
        write_json(root / "state.json", state)
        started = time.perf_counter()
        event = {"stage": name, "started_at": utc_now(), "source_bundle_sha256": code_hash}
        try:
            yield
            event["status"] = "completed"
        except BaseException as error:
            event.update(status="failed", error_type=type(error).__name__, error=str(error))
            raise
        finally:
            event["elapsed_seconds"] = time.perf_counter() - started
            event["finished_at"] = utc_now()
            append_jsonl(root / "stages.jsonl", event)

    try:
        from config.neo4jdb import get_db_manager
        from qa.prediction import HotpotKGPipeline
        graph = get_db_manager().graph
        with stage("local_preflight"):
            versions = graph.query("RETURN gds.version() AS gds, apoc.version() AS apoc")
            usage_reporting = graph.query(
                "SHOW SETTINGS YIELD name, value WHERE name = 'dbms.usage_report.enabled' RETURN value"
            )
            if usage_reporting[0]["value"] != "false":
                raise RuntimeError("Isolated database vendor usage reporting must be disabled")
            local_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with local_opener.open("http://127.0.0.1:11434/api/tags", timeout=15) as response:
                models = json.load(response)["models"]
            bge = next(row for row in models if row["name"] == "bge-m3:latest")
            pipeline = HotpotKGPipeline(
                dataset_path=str(DESTINATION / "questions.parquet"),
                output_dir=str(root / "qa"), ids_path=str(DESTINATION / "ids.json"),
            )
            ids = json.loads((DESTINATION / "ids.json").read_text(encoding="utf-8"))
            if list(pipeline.dataset["id"]) != ids:
                raise ValueError("QA input differs from frozen ID order")
            write_json(root / "local_preflight.json", {
                "passed": True, "question_count": len(ids), "versions": versions,
                "local_embedding_model": bge, "database_usage_reporting": False,
                "model_api_calls_for_this_preflight": 0,
            })
        if args.local_preflight_only:
            state.update(status="local_preflight_completed", full_reproduction_completed=False)
            return
        run_full_pipeline(root, graph, pipeline, stage, state)
        state.update(status="completed", full_reproduction_completed=True, finished_at=utc_now())
    except BaseException as error:
        state.update(status="failed", error_type=type(error).__name__, error=str(error), finished_at=utc_now())
        (root / "failure_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        stopped.set()
        monitor.join(timeout=5)
        write_json(root / "usage_summary.json", usage_summary(root))
        write_json(root / "state.json", state)


def run_full_pipeline(root, graph, pipeline, stage, state):
    """Identical build stage order to build.main, followed by the supplied QA."""
    from build.build_graph import KnowledgeGraphBuilder
    from build.build_index_and_community import IndexCommunityBuilder
    from build.build_chunk_index import ChunkIndexBuilder

    construction_started = time.perf_counter()
    with stage("graph_build"):
        checkpoint = root / "base_graph_completed.json"
        if checkpoint.exists():
            validate_graph_alignment(graph)
        else:
            builder = KnowledgeGraphBuilder()
            builder.entity_extractor.cache_dir = str(root / "cache/graph")
            Path(builder.entity_extractor.cache_dir).mkdir(parents=True, exist_ok=True)
            # This is the first real construction chunk, retained in the same
            # extraction cache and reused by the full original builder.
            first = builder.dataset_processor.process_data(builder.dataset_processor.dataset[0])
            first_hash = builder.entity_extractor._generate_cache_key(first["chunks"][0])
            with call_context(stage="graph_extraction", chunk_hash=first_hash, source_files=[first["filename"]]):
                extraction = builder.entity_extractor._process_single_chunk(first["chunks"][0])
            if not re.search(r'\("entity" : "(.+?)" : "(.+?)" : "(.+?)"\)', extraction):
                ledger = root / "provider_refusals.jsonl"
                known_refusal = ledger.exists() and any(
                    row.get("chunk_hash") == first_hash
                    and row.get("content_sha256") == hashlib.sha256(extraction.encode("utf-8")).hexdigest()
                    for row in (json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()
                                if line.strip())
                )
                if not known_refusal:
                    raise ValueError("First construction response does not match the original tuple codec")
            builder.process()
            alignment = validate_graph_alignment(graph)
            snapshot = graph_snapshot(graph)
            if snapshot["entities"] == 0:
                raise RuntimeError("Graph contains no extracted entities")
            with (root / "processed_documents.pkl").open("wb") as stream:
                pickle.dump(builder.processed_documents, stream)
            write_json(checkpoint, {
                "completed_at": utc_now(), "performance_stats": builder.performance_stats,
                "alignment": alignment, "graph": snapshot,
            })
    with stage("entity_index"):
        if not (root / "entity_index_completed.json").exists():
            index_builder = IndexCommunityBuilder()
            index_builder.process()
            write_json(root / "entity_index_completed.json", {
                "completed_at": utc_now(), "performance_stats": index_builder.performance_stats,
            })
    with stage("chunk_index"):
        if not (root / "chunk_index_completed.json").exists():
            chunk_builder = ChunkIndexBuilder()
            chunk_builder.process()
            write_json(root / "chunk_index_completed.json", {
                "completed_at": utc_now(), "performance_stats": chunk_builder.performance_stats,
            })
        # Neo4j populates new indexes asynchronously after creation returns.
        graph.query("CALL db.awaitIndexes(300)")
        snapshot = graph_snapshot(graph)
        for label in ("__Entity__", "__Chunk__"):
            counts = graph.query(
                f"MATCH (n:{label}) RETURN count(n) AS total, "
                "count(n.embedding) AS embedded, min(size(n.embedding)) AS min_dimension, "
                "max(size(n.embedding)) AS max_dimension"
            )[0]
            if counts["total"] != counts["embedded"] or counts["min_dimension"] != 1024 or counts["max_dimension"] != 1024:
                raise ValueError(f"Incomplete BGE-M3 index: {label}: {counts}")
        expected_indexes = {"entity", "chunk"}
        online = {row["name"] for row in snapshot["vector_indexes"] if row["state"] == "ONLINE"}
        if not expected_indexes.issubset(online):
            raise ValueError(f"Vector indexes are not online: {online}")
        write_json(root / "construction_summary.json", {
            "completed_at": utc_now(),
            "this_invocation_construction_seconds": time.perf_counter() - construction_started,
            "graph": snapshot, "alignment": validate_graph_alignment(graph),
        })
    with stage("qa_inference"):
        results = pipeline.run_on_dataset()
        if any(row["status"] != "completed" for row in results):
            raise RuntimeError("Some frozen questions failed after retries; inspect QA checkpoints")
    with stage("evaluation"):
        spec = importlib.util.spec_from_file_location("benchmark_evaluation", ROOT / "qa/benchmark_evaluate.py")
        evaluator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evaluator)
        ids = json.loads((DESTINATION / "ids.json").read_text(encoding="utf-8"))
        metrics = evaluator.evaluate(
            str(root / "qa/predictions.json"), str(DESTINATION / "gold.jsonl"), ids
        )
        if metrics["denominator"] != SAMPLE_SIZE or metrics["missing_prediction_ids"]:
            raise ValueError("Evaluation did not cover all frozen 1000 questions")
        write_json(root / "metrics.json", metrics)
        latencies = np.array([row["elapsed_seconds"] for row in results])
        summary = {
            "questions": SAMPLE_SIZE, "successful_questions": SAMPLE_SIZE,
            "mean_query_seconds": float(latencies.mean()),
            "p50_query_seconds": float(np.percentile(latencies, 50)),
            "p95_query_seconds": float(np.percentile(latencies, 95)),
            "early_two_hop_answers": sum(row["early_two_hop_answer"] for row in results),
            "final_graph": graph_snapshot(graph), "usage": usage_summary(root),
        }
        if summary["usage"]["observed_completed_interval_peak_concurrency"] > TEXT_MAX_WORKERS:
            raise RuntimeError(f"Observed model request concurrency exceeded {TEXT_MAX_WORKERS}")
        write_json(root / "final_summary.json", summary)


if __name__ == "__main__":
    main()
