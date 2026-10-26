"""Run with: python -m revision_openrouter.run --name pilot1 --limit 1."""
from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import time
import traceback

from hypergraphrag import HyperGraphRAG, QueryParam
from hypergraphrag.operate import chunking_by_token_size
from hypergraphrag.prompt import PROMPTS
from hypergraphrag.utils import EmbeddingFunc, LLMUnavailableError, compute_mdhash_id
from .transport import (
    BASE_URL, EMBED_DIGEST, EMBED_MODEL, ITEM, MODEL, PROVIDER,
    REASONING_EFFORT, TEMPERATURE, Transport,
    append_jsonl, file_hash, signature, write_json,
)
from .telemetry import (
    mark_phase, process_resources, resource_monitor, save_environment,
)
from .embedding_client import OVERFLOW_POLICY

BASE = Path(__file__).resolve().parents[1]
PROJECT = Path(os.environ["SHGRAPHRAG_DATA_ROOT"])
LOCK = PROJECT / "gpt4omini/datasets.lock.json"
RESPONSE_TYPE = (
    "A short answer for exact-match question answering. Return only a JSON object "
    'with one key, "answer", whose value is the concise answer string. '
    "Use the shortest complete entity name, date, number or yes/no answer as "
    "appropriate. Do not include explanations, citations, Markdown or reasoning. "
    'If the supplied evidence is insufficient, use {"answer": ""}.'
)
CONFIG = {
    "variant": "official_hybrid_OpenRouter_gpt4omini_BGE_M3_revision",
    "source_commit": "d587cdf8c3fe2be7719557f845324cb3a321f5e2",
    "model": MODEL, "provider": PROVIDER, "base_url": BASE_URL,
    "text_api": "chat_completions", "reasoning_effort": REASONING_EFFORT,
    "temperature": TEMPERATURE,
    "temperature_note": "Fixed at zero for this new controlled revision experiment",
    "answer_response_format": {"type": "json_object"},
    "cache_policy": (
        "initial_formal_cache_empty; recovery_may_import_verified_same_model_requests_"
        "and_vectors_with_original_usage_and_failed_attempt_history; no_Luna_or_pilot_import"),
    "text_network_transport": "aiohttp_proxy_client_with_TLS_failure_cleanup",
    "text_total_timeout_s": 240,
    "text_error_recovery": {
        "max_attempts_per_exact_request_invocation": 4,
        "retry_statuses": [408, 409, 425, 429, "5xx"],
        "structured_provider_error_in_http200": "same_explicit_status_retry_rules",
        "identity_checks_before_choice_error_retry": True,
        "model_prompt_cap_temperature_routing": "unchanged_between_attempts",
        "partial_error_text": "provenance_only_never_used_as_evidence_or_answer",
        "failure_artifacts": "cache/text_failures/",
        "per_attempt_id_and_retry_backoff": "api_calls.jsonl",
        "terminal_filter_length_missing_finish_policy": "unchanged",
    },
    "provider_refusal_policy": {
        "explicit_content_filter_or_refusal": "terminal_unavailable_output_no_retry",
        "initial_extraction": "retain_corpus_and_vector_no_facts_from_filtered_output",
        "glean_or_coverage": "stop_chunk_generation_keep_prior_complete_facts",
        "summary": "retain_existing_description_without_generated_summary",
        "query_extraction_or_answer": "empty_answer_scored_in_full_frozen_denominator",
        "partial_filtered_text": "provenance_only_never_used_as_evidence_or_answer",
    },
    "output_truncation_policy": {
        "finish_reason_length": "terminal_unavailable_output_no_retry",
        "output_token_caps": "unchanged",
        "initial_extraction": "retain_corpus_and_vector_no_facts_from_truncated_output",
        "glean_or_coverage": "stop_chunk_generation_keep_prior_complete_facts",
        "summary": "retain_existing_description_without_generated_summary",
        "query_extraction_or_answer": "empty_answer_scored_in_full_frozen_denominator",
        "partial_truncated_text": "provenance_only_never_used_as_evidence_or_answer",
    },
    "output_incomplete_policy": {
        "missing_or_null_finish_reason": "terminal_unavailable_output_no_retry",
        "cause": "unknown_not_classified_as_filtering_or_token_cap",
        "initial_extraction": "retain_corpus_and_vector_no_facts_from_incomplete_output",
        "glean_or_coverage": "stop_chunk_generation_keep_prior_complete_facts",
        "summary": "retain_existing_description_without_generated_summary",
        "query_extraction_or_answer": "empty_answer_scored_in_full_frozen_denominator",
        "partial_text": "provenance_only_never_used_as_evidence_or_answer",
    },
    "completed_dataset_resume": "preserve_index_predictions_summary_and_original_timing_no_new_queries",
    "malformed_answer_policy": "retain_raw_response_empty_answer_scored_full_denominator",
    "llm_seed": None, "sample_seed": 20261020,
    "embedding": EMBED_MODEL, "embedding_digest": EMBED_DIGEST,
    "embedding_dimension": 1024, "embedding_normalize": True,
    "embedding_truncate": False,
    "embedding_overflow_policy": dict(OVERFLOW_POLICY),
    "embedding_execution_options": {"num_ctx": 8192, "num_batch": 8192},
    "embedding_endpoint": "http://127.0.0.1:11436/api/embed",
    "embedding_runtime_scope": "dedicated_local_Ollama_process_port11436",
    "embedding_http_recovery": {
        "max_attempts_per_exact_request": 4,
        "retry_statuses": [400, 408, 409, 425, 429, "5xx"],
        "retry_transport_errors": True,
        "exact_payload_unchanged_between_retries": True,
        "explicit_context_overflow": "separate_audited_truncate_true_fallback",
        "deterministic_error_after_bound": "stop_with_original_raw_response",
    },
    "chunk_token_size": 1200, "chunk_overlap_token_size": 100,
    "tokenizer": "o200k_base via gpt-4o-mini, original source accounting",
    "entity_extract_max_gleaning": 2, "entity_summary_to_max_tokens": 500,
    "retrieval_mode": "hybrid", "top_k": 60,
    "context_tokens_per_section_per_branch": 4000,
    "response_type": RESPONSE_TYPE,
    "storage": "official_NetworkX_NanoVectorDB_JsonKV",
    "inference_scope": "global_union_of_selected_candidate_passages",
    "index_extraction_call_provenance": True,
    "generation_output_caps": {"extract": 8192, "query_extract": 8192, "glean": 8192,
                               "coverage": 64, "summary": 500, "answer": 512},
    "telemetry": {
        "resource_interval_s": 10, "gpu_interval_s": 30,
        "query_subphase_timing": True, "all_attempt_usage": True,
        "path_question_repository": "not_applicable_to_official_HyperGraphRAG",
    },
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def prepare(entry, root, limit):
    """Validate shared inputs without opening gold or support annotations."""
    files = {}
    for key in ("ids", "candidate_corpus"):
        info = entry["sources"][key]
        path = PROJECT / info["path"]
        if file_hash(path) != info["sha256"]:
            raise RuntimeError("Frozen data hash mismatch: " + key)
        files[key] = path
    info = entry["inference_queries"]
    queries_path = PROJECT / info["path"]
    if file_hash(queries_path) != info["sha256"]:
        raise RuntimeError("Query file hash mismatch")
    ids = read_json(files["ids"])
    queries = read_jsonl(queries_path)
    if len(ids) != 1000 or len(set(ids)) != 1000 or [q["id"] for q in queries] != ids:
        raise RuntimeError("Frozen queries are not exactly 1000 aligned IDs")
    if any(set(q) != {"id", "question"} for q in queries):
        raise RuntimeError("Unexpected inference query fields")
    corpus = read_jsonl(files["candidate_corpus"])
    if len(corpus) != entry["candidate_passage_count"]:
        raise RuntimeError("Corpus count mismatch")
    if len({p["passage_id"] for p in corpus}) != len(corpus):
        raise RuntimeError("Duplicate corpus IDs")
    selected_ids = ids[:limit]
    if limit < 1000:
        # Select pilot corpus at preparation only; retrieval has no query filter.
        info = entry["sources"]["question_passages_evaluation_only"]
        path = PROJECT / info["path"]
        if file_hash(path) != info["sha256"]:
            raise RuntimeError("Pilot membership hash mismatch")
        wanted = {pid for row in read_jsonl(path)
                  if row["question_id"] in set(selected_ids) for pid in row["passage_ids"]}
        corpus = [p for p in corpus if p["passage_id"] in wanted]
    manifest = {"dataset": entry["dataset"], "question_count": limit,
                "passage_count": len(corpus), "ids": selected_ids,
                "sample_seed": 20261020, "corpus_sha256": entry["sources"]["candidate_corpus"]["sha256"],
                "queries_sha256": entry["inference_queries"]["sha256"],
                "retrieval_filter": "none", "pilot": limit < 1000,
                "status": "new_revision_samples_not_historical_paper_samples"}
    write_json(root / "data_manifest.json", manifest)
    return queries[:limit], corpus, manifest


def provenance(root, workers):
    source = {}
    for directory in ("hypergraphrag", "revision_openrouter"):
        for path in sorted((BASE / directory).rglob("*.py")):
            if any(part in ("runs", "cache", "__pycache__") for part in path.parts):
                continue
            source[str(path.relative_to(BASE))] = file_hash(path)
    # The embedding implementation and JSON/hash utilities are shared read-only.
    source["revision/transport.py"] = file_hash(BASE / "revision/transport.py")
    source["revision_openrouter/launch.ps1"] = file_hash(
        BASE / "revision_openrouter/launch.ps1")
    source["monitoring/start_isolated_embedding.ps1"] = file_hash(
        BASE / "monitoring/start_isolated_embedding.ps1")
    git_head = subprocess.run(
        ["git", "-C", str(BASE), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True, encoding="utf-8", timeout=15).stdout.strip()
    pinned = read_json(BASE / "frozen_algorithm.json")
    for relative, expected in pinned.items():
        if file_hash(BASE / relative) != expected:
            raise RuntimeError("Pinned algorithm file changed: " + relative)
    source_patch = subprocess.run(
        ["git", "-C", str(BASE), "diff", "--", "hypergraphrag", ".gitignore"],
        check=True, capture_output=True, text=True, encoding="utf-8", timeout=15).stdout
    record = {"config": CONFIG, "source_sha256": source,
              "config_sha256": signature(CONFIG), "workers": workers,
              "git_head": git_head, "tracked_source_patch_sha256": signature(source_patch),
              "python": platform.python_version(), "platform": platform.platform(),
              "hash_seed": os.getenv("PYTHONHASHSEED"),
              "dependency_lock_sha256": file_hash(BASE / "revision/requirements.lock.txt"),
              "dataset_lock_sha256": file_hash(LOCK)}
    path = root / "provenance.json"
    if path.exists():
        old = read_json(path)
        if any(old.get(key) != record.get(key) for key in (
                "config", "source_sha256", "workers", "hash_seed",
                "dependency_lock_sha256", "dataset_lock_sha256")):
            raise RuntimeError("Source/config/environment changed: use a new run name")
    else:
        write_json(path, record)
        write_json(root / "prompts.json", PROMPTS)
        (root / "tracked_source.patch").write_text(source_patch, encoding="utf-8")
        for relative in source:
            destination = root / "source_snapshot" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(BASE / relative, destination)
        shutil.copy2(BASE / "revision/requirements.lock.txt",
                     root / "source_snapshot/requirements.lock.txt")
        shutil.copy2(LOCK, root / "datasets.lock.json")
    return record


@contextmanager
def campaign_lock(campaign):
    """Prevent duplicate paid workers for the same campaign, including resumes."""
    lock_path = campaign / ".worker.lock"
    with lock_path.open("a+b") as stream:
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


def make_rag(index, transport, workers):
    async def complete(*args, **kwargs):
        return await transport.complete(*args, **kwargs)
    async def embed(texts):
        return await transport.embed(texts)
    return HyperGraphRAG(
        working_dir=str(index), llm_model_name=MODEL, llm_model_func=complete,
        llm_model_max_async=workers, enable_llm_cache=False,
        embedding_func=EmbeddingFunc(1024, 8192, embed, concurrent_limit=1),
        embedding_func_max_async=1, embedding_batch_num=16,
        chunk_token_size=1200, chunk_overlap_token_size=100,
        entity_extract_max_gleaning=2, entity_summary_to_max_tokens=500,
        log_level="INFO",
    )


def document_text(passage):
    return passage["title"] + "\n" + passage["text"]


def parse_answer(text):
    value = text.strip()
    if value.startswith("```"):
        fenced = value.splitlines()
        if len(fenced) < 3 or not fenced[-1].startswith("```"):
            raise ValueError("answer_has_incomplete_markdown_fence")
        value = "\n".join(fenced[1:-1]).strip()
    try:
        data = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("answer_is_not_complete_JSON") from exc
    if not isinstance(data, dict) or set(data) != {"answer"} or not isinstance(data["answer"], str):
        raise ValueError("invalid_short_answer_schema")
    return data["answer"].strip()


def graph_statistics(rag):
    graph = rag.chunk_entity_relation_graph._graph
    roles = [data.get("role") for _, data in graph.nodes(data=True)]
    return {"full_docs": len(rag.full_docs._data), "chunks": len(rag.text_chunks._data),
            "entity_nodes": roles.count("entity"), "hyperedge_nodes": roles.count("hyperedge"),
            "incidence_edges": graph.number_of_edges(),
            "nodes_without_role": sum(role not in ("entity", "hyperedge") for role in roles)}


async def run_dataset(entry, campaign, transport, args, state):
    dataset_started = time.perf_counter()
    root = campaign / entry["dataset"]
    root.mkdir(parents=True, exist_ok=True)
    ITEM.set({"dataset": entry["dataset"]})
    mark_phase(campaign, entry["dataset"], "prepare_start")
    prepare_started = time.perf_counter()
    queries, corpus, manifest = prepare(entry, root, args.limit)
    timing = {"invocation_started_unix": time.time(),
              "prepare_wall_s": time.perf_counter() - prepare_started}
    mark_phase(campaign, entry["dataset"], "prepare_complete",
               elapsed_s=timing["prepare_wall_s"])
    state.pop("index", None)
    state.update(dataset=entry["dataset"], phase="index", expected_questions=len(queries),
                 expected_passages=len(corpus), completed_questions=0)
    pointer = root / "index_complete.json"
    summary_path = root / "summary.json"
    if pointer.exists() and summary_path.exists() and (root / "timing.json").exists():
        summary = read_json(summary_path)
        saved = [read_json(path) for path in (root / "predictions").glob("*.json")]
        expected = {query["id"]: query["question"] for query in queries}
        if (len(saved) == len(expected) and {row["id"] for row in saved} == set(expected)
                and summary.get("expected") == summary.get("completed") == len(expected)
                and all(row.get("status") == "completed" and row.get("model") == MODEL
                        and row.get("provider") == PROVIDER
                        and row.get("reasoning_effort") == REASONING_EFFORT
                        and row.get("question") == expected[row["id"]]
                        and signature(row.get("context")) == row.get("context_sha256")
                        for row in saved)):
            state.update(phase="completed_dataset_reused", completed_questions=len(expected),
                         index=read_json(pointer))
            mark_phase(campaign, entry["dataset"], "completed_dataset_reused",
                       completed_questions=len(expected), summary_sha256=file_hash(summary_path),
                       timing_sha256=file_hash(root / "timing.json"))
            return summary
    docs = [document_text(p) for p in corpus]
    doc_to_pid = {compute_mdhash_id(document.strip(), prefix="doc-"): p["passage_id"]
                  for document, p in zip(docs, corpus)}
    if len(doc_to_pid) != len(corpus):
        raise RuntimeError("Document content collision: preserve and resolve before indexing")
    write_json(root / "document_passage_map.json", doc_to_pid)
    expected_chunks = {}
    for document in docs:
        for chunk in chunking_by_token_size(
                document.strip(), 100, 1200, tiktoken_model="gpt-4o-mini"):
            expected_chunks[compute_mdhash_id(chunk["content"], prefix="chunk-")] = True
    state["expected_chunks"] = len(expected_chunks)
    reused_index = pointer.exists()
    if reused_index:
        index_info = read_json(pointer)
        rag = make_rag(root / index_info["directory"], transport, args.workers)
    else:
        # Fresh index transaction on every failed build; reuse response/vector
        # caches, never merge already-partial incidence weights a second time.
        name = "index_" + time.strftime("%Y%m%d_%H%M%S")
        index = root / name
        rag = make_rag(index, transport, args.workers)
        started = time.perf_counter()
        mark_phase(campaign, entry["dataset"], "construction_start", directory=name)
        before_calls, before_hits = transport.calls, transport.hits
        build_attempt = {"directory": name, "started_unix": time.time()}
        try:
            await rag.ainsert(docs)
            stats = graph_statistics(rag)
            if stats["full_docs"] != len(docs) or set(rag.text_chunks._data) != set(expected_chunks):
                raise RuntimeError("Incomplete full corpus/chunk coverage")
            if not stats["entity_nodes"] or not stats["hyperedge_nodes"] or stats["nodes_without_role"]:
                raise RuntimeError("Invalid hypergraph node roles/empty graph")
        except BaseException as exc:
            append_jsonl(root / "construction_attempts.jsonl", {
                **build_attempt, "status": "failed", "finished_unix": time.time(),
                "elapsed_s": time.perf_counter() - started,
                "error_type": type(exc).__name__,
                "new_llm_calls": transport.calls - before_calls,
                "cached_llm_calls": transport.hits - before_hits})
            raise
        index_info = {"directory": name, "elapsed_s": time.perf_counter() - started,
                      "new_llm_calls": transport.calls - before_calls,
                      "cached_llm_calls": transport.hits - before_hits, **stats}
        append_jsonl(root / "construction_attempts.jsonl", {
            **build_attempt, **index_info, "status": "completed",
            "finished_unix": time.time()})
        write_json(pointer, index_info)
        mark_phase(campaign, entry["dataset"], "construction_complete", **index_info)
    timing["completed_index_construction_s"] = index_info["elapsed_s"]
    timing["index_reused_this_invocation"] = reused_index
    write_json(root / "timing.json", timing)
    state.update(phase="query", index=index_info)
    if getattr(args, "index_only", False):
        state.update(phase="index_ready_query_not_started")
        mark_phase(campaign, entry["dataset"], "index_ready_query_not_started",
                   index=index_info)
        return None
    predictions = root / "predictions"
    predictions.mkdir(exist_ok=True)
    query_slots = asyncio.Semaphore(args.workers)
    query_phase_started = time.perf_counter()
    resumed_questions = 0
    mark_phase(campaign, entry["dataset"], "query_phase_start", workers=args.workers)
    async def predict(query):
        async with query_slots:
            await predict_one(query)
    async def predict_one(query):
        nonlocal resumed_questions
        ITEM.set({"dataset": entry["dataset"], "item_id": query["id"], "stage": "query"})
        path = predictions / (query["id"] + ".json")
        if path.exists() and read_json(path).get("status") == "completed":
            state["completed_questions"] += 1
            resumed_questions += 1
            return
        started = time.perf_counter()
        count_key = (entry["dataset"], query["id"])
        before_calls = transport.item_counts[(*count_key, "calls")]
        before_hits = transport.item_counts[(*count_key, "hits")]
        before_embed = transport.item_counts[(*count_key, "embedding_requests")]
        param = QueryParam(only_need_context=True, top_k=60,
                           max_token_for_text_unit=4000, max_token_for_global_context=4000,
                           max_token_for_local_context=4000)
        retrieval_started = time.perf_counter()
        refusal = None
        try:
            context = await rag.aquery(query["question"], param)
        except LLMUnavailableError as exc:
            context, refusal = None, exc
        retrieval_latency_s = time.perf_counter() - retrieval_started
        ordered_pids = []
        for branch in ("global", "local"):
            for chunk in param.retrieval_trace.get(branch, {}).get("chunks", []):
                pid = doc_to_pid.get(chunk["full_doc_id"])
                if not pid:
                    raise RuntimeError("Retrieved chunk has no frozen passage provenance")
                if pid not in ordered_pids:
                    ordered_pids.append(pid)
        # Same response prompt used by official kg_query, with its public
        # response_type option set to a short benchmark answer format.
        system = PROMPTS["rag_response"].format(
            context_data=context or "", response_type=RESPONSE_TYPE)
        answer_started = time.perf_counter()
        invalid_answer_schema = False
        if context is None or context == PROMPTS["fail_response"]:
            answer, raw = "", PROMPTS["fail_response"]
        else:
            try:
                raw = await transport.complete(query["question"], system_prompt=system,
                                               max_tokens=512)
                try:
                    answer = parse_answer(raw)
                except ValueError:
                    answer, invalid_answer_schema = "", True
            except LLMUnavailableError as exc:
                answer, raw, refusal = "", exc.raw_text, exc
        pred = {"id": query["id"], "question": query["question"],
                "status": "completed", "answer": answer, "raw_answer": raw,
                "model": MODEL, "provider": PROVIDER,
                "reasoning_effort": REASONING_EFFORT,
                "retrieved_passage_ids": ordered_pids,
                "retrieval_trace": param.retrieval_trace,
                "answer_outcome": (refusal.outcome if refusal else
                                   "invalid_answer_schema" if invalid_answer_schema else "completed"),
                "provider_refusal": (
                    {"reason": str(refusal), "request_sha256": refusal.request_sha256,
                     "response_id": refusal.response_id}
                    if refusal and refusal.outcome == "provider_filtered" else None),
                "unavailable_output": (
                    {"reason": str(refusal), "outcome": refusal.outcome,
                     "request_sha256": refusal.request_sha256,
                     "response_id": refusal.response_id} if refusal else None),
                "context": context, "context_sha256": signature(context),
                "latency_s": time.perf_counter() - started,
                "retrieval_latency_s": retrieval_latency_s,
                "answer_generation_latency_s": time.perf_counter() - answer_started,
                "embedding_requests": (
                    transport.item_counts[(*count_key, "embedding_requests")] - before_embed),
                "new_llm_calls": transport.item_counts[(*count_key, "calls")] - before_calls,
                "cached_llm_calls": transport.item_counts[(*count_key, "hits")] - before_hits}
        write_json(path, pred)
        state["completed_questions"] += 1
        write_json(campaign / "status.json", {**state, "time": time.time(),
                                             "llm_calls": transport.calls})
        print(f"PROGRESS {entry['dataset']} query "
              f"{state['completed_questions']}/{len(queries)}", flush=True)
    await asyncio.gather(*(predict(query) for query in queries))
    timing.update(
        query_phase_wall_s=time.perf_counter() - query_phase_started,
        completed_questions=len(queries), reused_completed_questions=resumed_questions,
        new_questions_this_invocation=len(queries) - resumed_questions)
    timing["query_phase_questions_per_s"] = (
        (len(queries) - resumed_questions) / timing["query_phase_wall_s"])
    mark_phase(campaign, entry["dataset"], "query_phase_complete",
               elapsed_s=timing["query_phase_wall_s"], completed_questions=len(queries))
    state.update(phase="evaluation")
    evaluation_started = time.perf_counter()
    from .evaluate import evaluate_run
    metrics = evaluate_run(root, PROJECT, entry)
    latencies = [read_json(path)["latency_s"] for path in predictions.glob("*.json")]
    latency = {"mean_s": statistics.mean(latencies),
               "p50_s": float(__import__("numpy").percentile(latencies, 50)),
               "p95_s": float(__import__("numpy").percentile(latencies, 95)),
               "concurrent_query_workers": args.workers,
               "note": "Shared provider/GPU contention; record cache hits separately"}
    metrics.update(index=index_info, latency=latency)
    timing.update(
        evaluation_wall_s=time.perf_counter() - evaluation_started,
        dataset_invocation_wall_s=time.perf_counter() - dataset_started)
    write_json(root / "timing.json", timing)
    mark_phase(campaign, entry["dataset"], "evaluation_complete",
               elapsed_s=timing["evaluation_wall_s"])
    write_json(root / "summary.json", metrics)
    return metrics


async def main_async(args):
    campaign = BASE / "revision_openrouter/runs" / args.name
    campaign.mkdir(parents=True, exist_ok=True)
    with campaign_lock(campaign):
        await execute_campaign(args, campaign)


async def execute_campaign(args, campaign):
    provenance(campaign, args.workers)
    execution = {"limit": args.limit, "datasets": args.datasets, "workers": args.workers}
    if (campaign / "execution.json").exists():
        if read_json(campaign / "execution.json") != execution:
            raise RuntimeError("Run arguments changed: use a new run name")
    else:
        write_json(campaign / "execution.json", execution)
    save_environment(campaign)
    invocation_started_unix = time.time()
    invocation_started = time.perf_counter()
    hardware = {"cpu": platform.processor(), "cpu_logical_count": os.cpu_count(),
                "wall_clock_note": "Shared API and local GPU; not an exclusive-load measurement"}
    if shutil.which("nvidia-smi"):
        probe = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used",
             "--format=csv,noheader"], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=15)
        hardware.update(gpu_probe_exit_code=probe.returncode,
                        gpu_probe=probe.stdout.strip())
    append_jsonl(campaign / "hardware_snapshots.jsonl", {
        **hardware, "time": time.time()})
    entries = {r["dataset"]: r for r in read_json(LOCK)["datasets"]}
    cache_only = getattr(args, "cache_only", False)
    index_only = getattr(args, "index_only", False)
    append_jsonl(campaign / "invocation_modes.jsonl", {
        "started_unix": invocation_started_unix,
        "cache_only": cache_only, "index_only": index_only,
        "text_network_enabled": not cache_only,
    })
    transport = Transport(campaign / "cache", campaign, args.workers,
                          cache_only=cache_only)
    state = {"status": "running", "phase": "preflight", "started_unix": time.time()}
    async def heartbeat():
        while True:
            write_json(campaign / "status.json", {
                **state, "updated_unix": time.time(), "llm_calls": transport.calls,
                "cache_hits": transport.hits, "embedding_calls": transport.embedding_calls,
                "provider_filtered_calls": transport.filtered_calls,
                "provider_filtered_cache_hits": transport.filtered_cache_hits,
                "output_truncated_calls": transport.truncated_calls,
                "output_truncated_cache_hits": transport.truncated_cache_hits,
                "output_incomplete_calls": transport.incomplete_calls,
                "output_incomplete_cache_hits": transport.incomplete_cache_hits,
                "initially_extracted_chunks": len(
                    transport.extracted_chunks.get(state.get("dataset"), set())),
                "initially_filtered_chunks": len(
                    transport.refused_chunks.get(state.get("dataset"), set())),
                "initially_truncated_chunks": len(
                    transport.truncated_chunks.get(state.get("dataset"), set())),
                "initially_incomplete_chunks": len(
                    transport.incomplete_chunks.get(state.get("dataset"), set())),
                "stages": dict(transport.stages)})
            await asyncio.sleep(20)
    beat = asyncio.create_task(heartbeat())
    monitor = asyncio.create_task(resource_monitor(campaign, state, transport))
    try:
        write_json(campaign / "embedding_contract.json", await transport.check_embedding())
        results = []
        for name in args.datasets:
            results.append(await run_dataset(entries[name], campaign, transport, args, state))
        if index_only:
            state.update(status="awaiting_query", phase="indexes_ready")
            write_json(campaign / "index_preparation.json", {
                "status": "indexes_ready_queries_incomplete",
                "cache_only": cache_only, "new_llm_calls": transport.calls,
                "cached_llm_calls": transport.hits,
                "datasets": {
                    name: read_json(campaign / name / "index_complete.json")
                    for name in args.datasets},
                "invocation_wall_s": time.perf_counter() - invocation_started,
                "completed_unix": time.time(),
            })
            return
        summary = {"status": "predictions_complete_pending_verification",
                   "full_benchmark": args.limit == 1000 and set(args.datasets) == set(entries),
                   "datasets": results,
                   "invocation_wall_s": time.perf_counter() - invocation_started,
                   "average_em": statistics.mean(r["primary_em"] for r in results),
                   "average_f1": statistics.mean(r["primary_f1"] for r in results),
                   "average_native_em": statistics.mean(r["native_alias_em"] for r in results),
                   "average_native_f1": statistics.mean(r["native_alias_f1"] for r in results)}
        write_json(campaign / "summary.json", summary)
        state.update(phase="verification")
        if set(args.datasets) == set(entries):
            from .verify import summarize
            verification = summarize(campaign)
            if args.limit == 1000 and verification["status"] != "verified_complete":
                raise RuntimeError("Full benchmark did not pass independent verification")
        summary["status"] = "completed"
        write_json(campaign / "summary.json", summary)
        state.update(status="completed", phase="completed")
        print(json.dumps(summary, ensure_ascii=True), flush=True)
    except BaseException as exc:
        state.update(status="failed", error_type=type(exc).__name__, error=str(exc)[:300])
        write_json(campaign / "failure.json", {**state, "traceback": traceback.format_exc()})
        raise
    finally:
        beat.cancel()
        monitor.cancel()
        await asyncio.gather(beat, monitor, return_exceptions=True)
        append_jsonl(campaign / "resource_samples.jsonl", {
            **process_resources(), "dataset": state.get("dataset"),
            "phase": state.get("phase"), "llm_peak_active": transport.peak_active_calls})
        append_jsonl(campaign / "campaign_attempts.jsonl", {
            "started_unix": invocation_started_unix, "finished_unix": time.time(),
            "elapsed_s": time.perf_counter() - invocation_started,
            "status": state["status"], "llm_calls": transport.calls,
            "cache_hits": transport.hits,
            "peak_llm_in_flight": transport.peak_active_calls})
        write_json(campaign / "status.json", {**state, "updated_unix": time.time(),
                                             "llm_calls": transport.calls,
                                             "cache_hits": transport.hits,
                                             "stages": dict(transport.stages)})
        await transport.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--cache-only", action="store_true",
                        help="Never submit text requests; stop on any exact cache miss.")
    parser.add_argument("--index-only", action="store_true",
                        help="Commit indexes and return before starting unfinished queries.")
    parser.add_argument("--datasets", nargs="+", default=["hotpotqa", "2wiki", "musique"],
                        choices=["hotpotqa", "2wiki", "musique"])
    args = parser.parse_args()
    if not 0 < args.limit <= 1000 or not 0 < args.workers <= 32:
        parser.error("Expected 1..1000 questions and 1..32 workers")
    if not re_safe_name(args.name):
        parser.error("Run name must contain only letters, digits, underscores or hyphens")
    asyncio.run(main_async(args))


def re_safe_name(value):
    return bool(value) and all(c.isascii() and (c.isalnum() or c in "_-") for c in value)


if __name__ == "__main__":
    main()
