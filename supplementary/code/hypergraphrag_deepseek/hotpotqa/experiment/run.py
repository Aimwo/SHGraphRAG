"""Full native HyperGraphRAG algorithm with the explicitly selected backends."""
from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import sys
import time
import uuid

from .common import (
    ITEM, ROOT, append_jsonl, file_hash, file_inventory, read_json,
    read_jsonl, signature, write_json,
)
from .config import load_config, read_authorized_key
from .prepare import validate_inputs

os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(ROOT / "tokenizer_cache"))
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="backslashreplace")

from hypergraphrag import HyperGraphRAG, QueryParam
from hypergraphrag.operate import chunking_by_token_size
from hypergraphrag.prompt import PROMPTS
from hypergraphrag.utils import EmbeddingFunc, LLMUnavailableError, compute_mdhash_id
from .transport import ProviderError
from .summary_policy import SummaryTransport as Transport, POLICY as SUMMARY_POLICY
from .telemetry import mark_phase, resource_monitor, save_environment

DATASET = load_config(ready=False)["dataset_id"]

RESPONSE_TYPE = (
    "A short answer for exact-match question answering. Return only a JSON object "
    'with one key, "answer", whose value is the concise answer string. '
    "Use the shortest complete entity name, date, number or yes/no answer as appropriate. "
    "Do not include explanations, citations, Markdown or reasoning. "
    'If the supplied evidence is insufficient, use {"answer": ""}.'
)
METHOD = {
    "official_commit": "d587cdf8c3fe2be7719557f845324cb3a321f5e2",
    "algorithm": "HyperGraphRAG native tuple extraction and hybrid entity/hyperedge retrieval",
    "storage": "NetworkXStorage/NanoVectorDBStorage/JsonKVStorage",
    "chunk_token_size": 1200, "chunk_overlap_token_size": 100,
    "tokenizer": "gpt-4o-mini o200k_base (token accounting only; no GPT model calls)",
    "entity_extract_max_gleaning": 2, "entity_summary_to_max_tokens": 500,
    "mode": "hybrid", "top_k": 60,
    "max_token_for_text_unit": 4000,
    "max_token_for_global_context": 4000, "max_token_for_local_context": 4000,
    "response_type": RESPONSE_TYPE,
    "path_question_repository": {"applicable": False, "count": 0,
                                 "reason": "Not a component of this native HyperGraphRAG implementation"},
    "corpus_scope": "Global union of selected candidate paragraphs; no question-specific filter",
    "summary_backend_adaptation": SUMMARY_POLICY,
    "initial_model_cache": "Exact text requests retained; identical-input BGE vectors reused where available; changed summary descriptions re-embedded",
}


@contextmanager
def worker_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".worker.lock").open("a+b") as stream:
        stream.seek(0)
        if not stream.read(1):
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def code_files():
    return [ROOT / "launch.ps1", *[
        p for folder in ("hypergraphrag", "experiment") for p in (ROOT / folder).rglob("*")
        if p.is_file() and p.suffix in (".py", ".json") and "__pycache__" not in p.parts]]


def contract(config):
    original = read_json(ROOT / "provenance/source.json")
    if file_hash(ROOT / "provenance/algorithm.patch") != original["patch_sha256"]:
        raise ValueError("Captured upstream working-tree patch changed")
    for relative, expected in original["source_sha256"].items():
        if file_hash(ROOT / relative) != expected:
            raise ValueError("Algorithm source differs from captured source: " + relative)
    return {
        "config": config, "method": METHOD, "inputs": read_json(ROOT / "inputs/manifest.json"),
        "source_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(code_files())},
        "source_provenance": original,
        "dependency_lock_sha256": file_hash(ROOT / "provenance/requirements.lock.txt"),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "capacity_recovery_import": None,  # Public fresh run; no private caches imported.
    }


def store_provenance(run, config):
    record = contract(config)
    path = run / "provenance.json"
    if path.exists():
        if read_json(path) != record:
            raise ValueError("Source, input or configuration changed; preserve this run and use a new one")
    else:
        write_json(path, record)
        write_json(run / "prompts.json", PROMPTS)
        for source in code_files():
            target = run / "source_snapshot" / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    save_environment(run)
    return signature(record)


def make_rag(index, transport):
    async def complete(*args, **kwargs):
        return await transport.complete(*args, **kwargs)

    async def embed(texts):
        return await transport.embed(texts)

    return HyperGraphRAG(
        working_dir=str(index), llm_model_name=transport.config["model"],
        llm_model_func=complete, llm_model_max_async=transport.config["max_in_flight"],
        enable_llm_cache=False,
        embedding_func=EmbeddingFunc(
            1024, 8192, embed, concurrent_limit=transport.config["embedding"]["max_in_flight"]),
        embedding_func_max_async=transport.config["embedding"]["max_in_flight"],
        embedding_batch_num=transport.config["embedding"]["batch_size"],
        chunk_token_size=1200, chunk_overlap_token_size=100,
        entity_extract_max_gleaning=2, entity_summary_to_max_tokens=500,
        log_level="INFO",
    )


def graph_statistics(rag):
    graph = rag.chunk_entity_relation_graph._graph
    roles = [data.get("role") for _, data in graph.nodes(data=True)]
    return {"full_docs": len(rag.full_docs._data), "chunks": len(rag.text_chunks._data),
            "entity_nodes": roles.count("entity"), "hyperedge_nodes": roles.count("hyperedge"),
            "incidence_edges": graph.number_of_edges(),
            "nodes_without_role": sum(r not in ("entity", "hyperedge") for r in roles)}


def parse_answer(raw):
    value = raw.strip()
    if value.startswith("```") and "\n" in value:
        value = value.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        data = json.loads(value)
    except ValueError as exc:
        raise ValueError("invalid_short_answer_JSON") from exc
    if not isinstance(data, dict) or set(data) != {"answer"} or not isinstance(data["answer"], str):
        raise ValueError("invalid_short_answer_schema")
    return data["answer"].strip()


def corpus_plan(corpus):
    mapping = defaultdict(list)
    chunks = set()
    for passage in corpus:
        document = passage["text"].strip()
        mapping[compute_mdhash_id(document, prefix="doc-")].append(passage["id"])
        for chunk in chunking_by_token_size(
                document, 100, 1200, tiktoken_model="gpt-4o-mini"):
            chunks.add(compute_mdhash_id(chunk["content"], prefix="chunk-"))
    return dict(mapping), chunks


async def build(run, transport, corpus, state):
    ITEM.set({"dataset": DATASET, "stage": "construction"})
    mapping, expected_chunks = corpus_plan(corpus)
    write_json(run / "document_passage_map.json", mapping)
    write_json(run / "expected_chunks.json", sorted(expected_chunks))
    pointer = run / "index_complete.json"
    if pointer.exists():
        information = read_json(pointer)
        if (information["scope_sha256"] != transport.scope_hash
                or information["files"] != file_inventory(run / information["directory"])):
            raise ValueError("Previously completed index identity or file hashes changed")
        rag = make_rag(run / information["directory"], transport)
        if (set(rag.text_chunks._data) != expected_chunks
                or graph_statistics(rag) != information["graph"]):
            raise ValueError("Previously completed index changed")
        return rag, mapping, information
    index = run / ("index_" + str(time.time_ns()))
    rag = make_rag(index, transport)
    ITEM.set({"dataset": DATASET, "stage": "construction", "index_directory": index.name})
    started, started_unix = time.perf_counter(), time.time()
    before = (transport.calls, transport.hits, transport.embedding_calls)
    mark_phase(run, DATASET, "construction_started",
               paragraphs=len(corpus), native_unique_stripped_docs=len(mapping),
               expected_chunks=len(expected_chunks))
    outcome = {"directory": index.name, "started_unix": started_unix,
               "scope_sha256": transport.scope_hash}
    try:
        await rag.ainsert([p["text"] for p in corpus])
        stats = graph_statistics(rag)
        if stats["full_docs"] != len(mapping) or set(rag.text_chunks._data) != expected_chunks:
            raise ValueError("Incomplete corpus or chunk coverage")
        if not stats["entity_nodes"] or not stats["hyperedge_nodes"] or stats["nodes_without_role"]:
            raise ValueError("Empty graph or invalid hypergraph roles")
        outcome.update(status="completed", graph=stats, files=file_inventory(index))
    except BaseException as exc:
        outcome.update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        outcome.update(finished_unix=time.time(), wall_s=time.perf_counter() - started,
                       text_physical_attempts=transport.calls - before[0],
                       text_cache_hits=transport.hits - before[1],
                       embedding_physical_attempts=transport.embedding_calls - before[2])
        append_jsonl(run / "construction_attempts.jsonl", outcome)
        mark_phase(run, DATASET, "construction_finished", **outcome)
    write_json(pointer, outcome)
    state["graph"] = outcome["graph"]
    return rag, mapping, outcome


def validate_prediction(result, query, config, provenance_sha256, index_sha256, passage_ids):
    expected = {**query, "provider": "deepseek-relay", "model": config["model"],
                "credential_id": config["credential_id"],
                "provenance_sha256": provenance_sha256, "index_sha256": index_sha256}
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError("Saved prediction input/model/index identity changed")
    if result.get("context_sha256") != signature(result.get("context")):
        raise ValueError("Saved prediction context hash changed")
    if result.get("status") not in ("completed", "failed", "cancelled"):
        raise ValueError("Unknown prediction processing status")
    if not isinstance(result.get("answer"), str):
        raise ValueError("Prediction answer must be a string")
    if not set(result.get("retrieved_passage_ids", [])) <= passage_ids:
        raise ValueError("Prediction retrieved an unfrozen passage")


async def predict_one(query, rag, transport, doc_map, run, *,
                      provenance_sha256, index_sha256):
    context_token = ITEM.set({"dataset": DATASET, "item_id": query["id"], "stage": "query"})
    started, started_unix = time.perf_counter(), time.time()
    count_key = (DATASET, query["id"])
    before = {k: transport.item_counts[(*count_key, k)] for k in ("calls", "hits", "embedding_requests")}
    param = QueryParam(only_need_context=True, mode="hybrid", top_k=60,
                       max_token_for_text_unit=4000, max_token_for_global_context=4000,
                       max_token_for_local_context=4000)
    context, raw, answer, outcome, status = None, "", "", "completed", "completed"
    retrieved, timings, error, fatal = [], {}, None, None
    current_stage, unavailable_raw_text = "retrieval", ""
    attempt_id = uuid.uuid4().hex
    try:
        step = time.perf_counter()
        try:
            context = await rag.aquery(query["question"], param)
        finally:
            timings["retrieval_s"] = time.perf_counter() - step
        for branch in ("global", "local"):
            for chunk in param.retrieval_trace.get(branch, {}).get("chunks", []):
                if chunk.get("full_doc_id") not in doc_map:
                    raise ValueError("Retrieved chunk lacks frozen passage provenance")
                for pid in doc_map[chunk["full_doc_id"]]:
                    if pid not in retrieved:
                        retrieved.append(pid)
        current_stage = "answer_generation"
        step = time.perf_counter()
        try:
            if context is None or context == PROMPTS["fail_response"]:
                outcome = "no_retrieval_context"
            else:
                system = PROMPTS["rag_response"].format(context_data=context, response_type=RESPONSE_TYPE)
                raw = await transport.complete(query["question"], system_prompt=system)
                try:
                    answer = parse_answer(raw)
                except ValueError:
                    outcome = "invalid_answer_schema"
        finally:
            timings["answer_generation_s"] = time.perf_counter() - step
    except LLMUnavailableError as exc:
        outcome, unavailable_raw_text = exc.outcome, exc.raw_text
        if current_stage == "answer_generation":
            raw = exc.raw_text
        error = {"type": type(exc).__name__, "request_sha256": exc.request_sha256,
                 "response_id": exc.response_id, "stage": current_stage}
    except ProviderError as exc:
        status, outcome, error = "failed", "request_failure", {
            "type": type(exc).__name__, "code": str(exc), "stage": current_stage}
        if str(exc) in ("http_401", "http_402", "http_403", "returned_model_not_authorized"):
            fatal = exc
    except BaseException as exc:
        status = "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed"
        outcome, error, fatal = "interrupted_or_internal_error", {
            "type": type(exc).__name__, "stage": current_stage}, exc
    finally:
        ITEM.reset(context_token)
    result = {
        **query, "status": status, "answer": answer, "raw_answer": transport.redact(raw),
        "unavailable_raw_text": transport.redact(unavailable_raw_text),
        "answer_outcome": outcome, "error": error,
        "model": transport.config["model"], "provider": "deepseek-relay",
        "credential_id": transport.config["credential_id"],
        "provenance_sha256": provenance_sha256, "index_sha256": index_sha256,
        "query_attempt_id": attempt_id,
        "context": context, "context_sha256": signature(context),
        "retrieved_passage_ids": retrieved, "retrieval_trace": param.retrieval_trace,
        "started_unix": started_unix, "finished_unix": time.time(),
        "latency_s": time.perf_counter() - started, **timings,
        **{k: transport.item_counts[(*count_key, k)] - v for k, v in before.items()},
    }
    artifact = run / "query_attempts" / query["id"] / (attempt_id + ".json")
    write_json(artifact, result)
    append_jsonl(run / "query_attempts.jsonl", {
        "id": query["id"], "query_attempt_id": attempt_id,
        "status": status, "answer_outcome": outcome, "latency_s": result["latency_s"],
        "artifact": artifact.relative_to(run).as_posix(), "artifact_sha256": file_hash(artifact)})
    write_json(run / "predictions" / (query["id"] + ".json"), result)
    if fatal is not None:
        raise fatal
    return result


async def execute(config, run):
    validate_inputs()
    provenance_sha256 = store_provenance(run, config)
    # No credential is opened before input/config/source validation and the worker lock.
    transport = Transport(config, run / "cache", run, key=read_authorized_key(config))
    started, started_unix = time.perf_counter(), time.time()
    state = {"status": "running", "dataset": DATASET, "phase": "construction",
             "completed_questions": 0, "started_unix": started_unix}
    write_json(run / "status.json", {**state, "updated_unix": time.time()})
    monitor = asyncio.create_task(resource_monitor(run, state, transport))
    try:
        queries = read_jsonl(ROOT / "inputs/queries.jsonl")
        corpus = read_jsonl(ROOT / "inputs/candidate_corpus.jsonl")
        rag, mapping, _index = await build(run, transport, corpus, state)
        index_sha256 = signature(_index["files"])
        passage_ids = {p["id"] for p in corpus}
        state["phase"] = "query"
        mark_phase(run, DATASET, "query_started", questions=len(queries))
        query_started = time.perf_counter()
        slots = asyncio.Semaphore(config["max_in_flight"])
        reused = 0

        async def predict(query):
            nonlocal reused
            async with slots:
                path = run / "predictions" / (query["id"] + ".json")
                previous = read_json(path) if path.exists() else None
                if previous is not None:
                    validate_prediction(previous, query, config, provenance_sha256,
                                        index_sha256, passage_ids)
                if previous is not None and previous["status"] == "completed":
                    reused += 1
                else:
                    await predict_one(query, rag, transport, mapping, run,
                                      provenance_sha256=provenance_sha256,
                                      index_sha256=index_sha256)
                state["completed_questions"] += 1
                write_json(run / "status.json", {**state, "updated_unix": time.time()})
                if state["completed_questions"] % 10 == 0:
                    print("QUERY", state["completed_questions"], "/", len(queries), flush=True)

        query_status = "failed_or_interrupted"
        try:
            async with asyncio.TaskGroup() as group:
                for query in queries:
                    group.create_task(predict(query))
            query_status = "completed"
        finally:
            mark_phase(run, DATASET, "query_finished", wall_s=time.perf_counter() - query_started,
                       questions=len(queries), completed_questions=state["completed_questions"],
                       status=query_status, reused_completed_predictions=reused)
        state["phase"] = "evaluation"
        from .evaluate import evaluate
        metrics = evaluate(run)
        write_json(run / "summary.json", {
            "status": "predictions_complete_pending_verification", "metrics": metrics,
            "invocation_wall_s": time.perf_counter() - started})
        state.update(status="predictions_complete_pending_verification", phase="verification")
    except BaseException as exc:
        state.update(status="failed", error_type=type(exc).__name__)
        write_json(run / "failure.json", transport.redact({
            **state, "error": str(exc),
            "note": "Inspect physical request artifacts; no provider switch or new-key fallback"}))
        raise
    finally:
        monitor.cancel()
        await asyncio.gather(monitor, return_exceptions=True)
        # Native as_completed extraction may leave sibling tasks alive on error.
        # Drain them while clients and the credential are still valid so every
        # started physical attempt writes its own actual final evidence.
        pending = {task for task in asyncio.all_tasks()
                   if task is not asyncio.current_task() and not task.done()}
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        append_jsonl(run / "campaign_attempts.jsonl", {
            "started_unix": started_unix, "finished_unix": time.time(),
            "wall_s": time.perf_counter() - started, "status": state["status"],
            "text_physical_attempts": transport.calls, "embedding_physical_attempts": transport.embedding_calls,
            "text_cache_hits": transport.hits, "peak_all_requests": transport.peak_requests,
            "peak_text_requests": transport.peak_active_calls})
        write_json(run / "status.json", {**state, "updated_unix": time.time()})
        await transport.close()
    from .verify import verify
    try:
        verification = verify(run)
        if verification["status"] != "verified_complete":
            raise RuntimeError("Full experiment did not pass independent completion verification")
    except BaseException as exc:
        state.update(status="verification_failed", phase="verification",
                     error_type=type(exc).__name__)
        write_json(run / "status.json", {**state, "updated_unix": time.time()})
        raise
    summary = read_json(run / "summary.json")
    final_status = ("completed_with_failed_questions" if verification["missing_or_failed_predictions"]
                    else "completed")
    summary.update(status=final_status, verification=verification)
    write_json(run / "summary.json", summary)
    state.update(status=final_status, phase="completed")
    write_json(run / "status.json", {**state, "updated_unix": time.time()})
    print(json.dumps(summary, ensure_ascii=True), flush=True)


def check():
    inputs = validate_inputs()
    config = load_config(ready=False)
    blockers = []
    api_paused = False
    from .capacity_recovery import require_model_calls_allowed
    try:
        require_model_calls_allowed()
    except RuntimeError as exc:
        api_paused = True
        blockers.append(str(exc))
    try:
        load_config()
    except (ValueError, RuntimeError) as exc:
        blockers.append(str(exc))
    try:
        provenance_sha256 = signature(contract(config))
    except (ValueError, OSError) as exc:
        provenance_sha256 = None
        blockers.append(str(exc))
    report = {
        "status": ("api_calls_paused_by_user" if api_paused else
                   "ready_to_run" if not blockers else "awaiting_user_parameters"),
        "input_validation": inputs, "provider": config["provider"],
        "model": config["model"], "base_url": config["base_url"],
        "embedding": config["embedding"], "total_concurrency": config["max_in_flight"],
        "blockers": blockers, "model_api_requests": 0,
        "credential_loaded_by_check": False, "full_experiment_completed": False,
        "provenance_sha256": provenance_sha256,
        "method": METHOD,
    }
    write_json(ROOT / "audit/readiness.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        print(json.dumps(check(), ensure_ascii=True, indent=2))
        return
    config = load_config()  # Reject missing destinations before credentials or network.
    from .capacity_recovery import require_model_calls_allowed
    require_model_calls_allowed()
    run = ROOT / "run"
    with worker_lock(run):
        asyncio.run(execute(config, run))


if __name__ == "__main__":
    main()
