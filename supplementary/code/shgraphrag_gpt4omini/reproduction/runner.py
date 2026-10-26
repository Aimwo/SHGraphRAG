"""Checkpointed SHGraphRAG construction, global retrieval, QA, and evaluation."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time

import numpy as np

from . import prompts
from .client import (BASE_URL, EFFORT, MODEL, PROVIDER, RUNTIME_DIRECTORY,
                     ResponsesClient, append_jsonl, canonical, parse_json,
                     signature, write_json)
from .data import DATASETS, prepare, read_jsonl, sha256
from .embedding import Embeddings, VectorIndex
from .entity_resolution import resolve as resolve_entities
from .evaluate import evaluate_run
from .graph import (assemble, entity_key, iter_paths, parse_json_extraction, path_prompt_item,
                    path_statistics)
from .telemetry import RunTelemetry


ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = ROOT / RUNTIME_DIRECTORY
PATH_BATCH = 32
METHOD_VERSION = "source_alias_merge_global_graph_v2"


def parallel_work(items, worker, *, workers: int, on_done):
    """Bound pending work and preserve completed artifacts on interruption."""
    iterator = iter(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = set()
        exhausted = False
        while pending or not exhausted:
            while not exhausted and len(pending) < workers:
                try:
                    pending.add(pool.submit(worker, next(iterator)))
                except StopIteration:
                    exhausted = True
            if not pending:
                break
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                on_done(future.result())


def batches(items, n):
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


class Runner:
    def __init__(self, root: Path, workers: int):
        self.root = root
        self.workers = workers
        self.client = ResponsesClient(root, concurrency=workers, cache_root=CACHE_ROOT)
        self.embed = None
        self.graph = None
        self.indexes = {}
        self.progress("initialized", model=MODEL, reasoning_effort=EFFORT)

    def embeddings(self):
        if self.embed is None:
            self.embed = Embeddings(CACHE_ROOT, run_id=str(self.root))
        return self.embed

    def progress(self, stage, **fields):
        record = {"stage": stage, "time_unix": time.time(), **fields}
        append_jsonl(self.root / "progress.jsonl", record)
        write_json(self.root / "status.json", record)
        print(canonical(record), flush=True)

    @contextmanager
    def phase(self, name):
        self.progress(name, status="running")
        started = time.perf_counter()
        try:
            yield
        except Exception as error:
            self.progress(name, status="failed", error_type=type(error).__name__,
                          error_category=getattr(error, "category", None))
            raise
        else:
            elapsed = time.perf_counter() - started
            append_jsonl(self.root / "timing.jsonl", {
                "stage": name, "elapsed_s": elapsed, "completed_unix": time.time()})
            self.progress(name, status="completed", elapsed_s=elapsed)

    def structured(self, stage, item_id, system, payload, validator,
                   max_tokens=4096, schema=None):
        prompt = canonical(payload)
        for repair in range(3):
            response = self.client.call(
                stage=stage if repair == 0 else stage + "_repair",
                item_id=item_id, system=system, prompt=prompt,
                max_tokens=max_tokens,
                response_schema=schema if schema is not None else prompts.SCHEMAS[stage])
            try:
                result = parse_json(response["text"])
                validator(result)
                return result, response
            except (ValueError, TypeError, KeyError, AssertionError) as error:
                append_jsonl(self.root / "schema_validation.jsonl", {
                    "stage": stage, "item_id": item_id, "attempt": repair + 1,
                    "request_sha256": response["request_sha256"],
                    "error": str(error), "time_unix": time.time()})
                if repair == 2:
                    raise ValueError(f"{stage}: structured output remained invalid")
                prompt = (canonical(payload) + "\nThe previous output did not match "
                          "the required complete JSON schema. Return a full corrected "
                          "replacement, with no extra text. Error: " + str(error)
                          + "\nPrevious output:\n"
                          + response["text"])
        raise AssertionError("unreachable")

    def extract(self):
        corpus = read_jsonl(self.root / "inputs/corpus.jsonl")
        completed = 0
        def one(passage):
            pid = passage["passage_id"]
            destination = self.root / "extractions" / f"{pid}.json"
            expected = signature([prompts.EXTRACT_SYSTEM,
                                  prompts.extraction_input(passage)])
            if destination.exists():
                item = json.loads(destination.read_text(encoding="utf-8"))
                if item["input_fingerprint"] != expected:
                    raise RuntimeError("Existing extraction has a different prompt")
                return pid
            prompt = prompts.extraction_input(passage)
            endpoint_completion = None
            for repair in range(3):
                response = self.client.call(
                    stage="extraction" if not repair else "extraction_repair",
                    item_id=pid, system=prompts.EXTRACT_SYSTEM,
                    prompt=prompt, max_tokens=8192,
                    response_schema=prompts.EXTRACT_SCHEMA)
                try:
                    extracted = parse_json_extraction(parse_json(response["text"]))
                    break
                except (ValueError, TypeError, KeyError) as error:
                    append_jsonl(self.root / "schema_validation.jsonl", {
                        "stage": "extraction", "item_id": pid, "attempt": repair + 1,
                        "request_sha256": response["request_sha256"],
                        "error": str(error), "time_unix": time.time()})
                    if repair == 2:
                        if str(error).startswith("Relationship endpoint absent from entity records:"):
                            extracted, endpoint_completion, response = (
                                self.complete_endpoints(passage, response))
                            break
                        raise ValueError(f"Extraction schema failed: {pid}") from error
                    prompt = (prompts.extraction_input(passage) +
                              "\nRepair the previous output to satisfy ALL JSON "
                              "schema rules. Every relationship endpoint must "
                              "have an entity record. Add complete entity records "
                              "for EVERY missing endpoint named in the error. "
                              "Keep all valid records. Return the FULL corrected "
                              "JSON object. Error: " + str(error)
                              + "\nPrevious output:\n" + response["text"])
            write_json(destination, {"passage_id": pid, "input_fingerprint": expected,
                                    "request_sha256": response["request_sha256"],
                                    "extracted": extracted,
                                    **({"endpoint_completion": endpoint_completion}
                                       if endpoint_completion else {})})
            return pid
        def done(_):
            nonlocal completed
            completed += 1
            if completed % 10 == 0 or completed == len(corpus):
                self.progress("extraction", completed=completed, total=len(corpus))
        with self.phase("extraction"):
            parallel_work(corpus, one, workers=self.workers, on_done=done)

    def complete_endpoints(self, passage, base_response):
        """Fixed-name schema completion; accepted graph records stay unchanged."""
        original = parse_json(base_response["text"])
        known = {entity_key(e["name"]) for e in original["entities"]}
        names = sorted({r[field] for r in original["relationships"]
                        for field in ("source", "target")
                        if entity_key(r[field]) not in known})
        if not names:
            raise ValueError("Endpoint completion requires missing endpoint names")
        schema = prompts.obj({"records": prompts.obj({
            name: prompts.obj({"type": prompts.STR, "description": prompts.STR})
            for name in names})})
        payload = {
            "passage": {"title": passage["title"], "text": passage["text"]},
            "required_names": names, "existing_entities": original["entities"],
            "relationships": original["relationships"]}
        def validate(value):
            records = value["records"]
            if set(records) != set(names) or any(
                    not all(isinstance(record.get(k), str) and record[k].strip()
                            for k in ("type", "description"))
                    for record in records.values()):
                raise ValueError("Incomplete fixed-name endpoint fields")
        value, response = self.structured(
            "extraction_endpoint_completion", passage["passage_id"],
            prompts.ENDPOINT_SYSTEM, payload, validate, max_tokens=8192,
            schema=schema)
        merged = {
            "entities": original["entities"] + [
                {"name": name, **value["records"][name]} for name in names],
            "relationships": original["relationships"],
        }
        parsed = parse_json_extraction(merged)
        proof = {
            "codec": "fixed_missing_endpoint_records_v1",
            "base_request_sha256": base_response["request_sha256"],
            "base_response_id": base_response.get("response_id"),
            "required_names": names, "records": value["records"],
            "response_request_sha256": response["request_sha256"],
            "accepted_existing_records": "unchanged",
        }
        return parsed, proof, response

    def build_graph(self):
        with self.phase("graph"):
            corpus = read_jsonl(self.root / "inputs/corpus.jsonl")
            extracted = {}
            for passage in corpus:
                pid = passage["passage_id"]
                item = json.loads((self.root / "extractions" / f"{pid}.json")
                                  .read_text(encoding="utf-8"))
                extracted[pid] = item["extracted"]
            groups = json.loads((self.root / "inputs/construction_groups.json")
                                .read_text(encoding="utf-8"))
            with self.phase("entity_merge"):
                extracted = resolve_entities(self, extracted, groups,
                                             parallel_work, batches)
            self.graph = assemble(corpus, extracted)
            self.graph["stats"]["identity"] = (
                "normalized_names_with_BGE_candidates_and_LLM_confirmed_aliases")
            self.graph["stats"].update(path_statistics(self.graph))
            write_json(self.root / "graph.json", self.graph)
            write_json(self.root / "graph_stats.json", self.graph["stats"])
            self.progress("graph_counts", **self.graph["stats"])

    def load_graph(self):
        if self.graph is None:
            self.graph = json.loads((self.root / "graph.json").read_text(encoding="utf-8"))
        return self.graph

    def path_questions(self):
        graph = self.load_graph()
        total = graph["stats"]["two_edge_paths"]
        vector_bytes = total * 2 * 1024 * 4
        available = shutil.disk_usage(self.root).free
        if vector_bytes + 2 * 1024 ** 3 > available:
            write_json(self.root / "capacity_blocker.json", {
                "paths": total, "minimum_vector_bytes": vector_bytes,
                "available_bytes": available, "reserve_bytes": 2 * 1024 ** 3,
                "note": "No path cap or method change was applied.",
            })
            raise RuntimeError("The complete path index exceeds available worktree disk space")
        completed = 0
        def one(batch):
            digest = signature([p["id"] for p in batch])
            path = self.root / "path_batches" / f"{digest}.json"
            expected = signature([prompts.PATH_SYSTEM,
                                  [path_prompt_item(p, graph) for p in batch]])
            if path.exists():
                item = json.loads(path.read_text(encoding="utf-8"))
                if item["input_fingerprint"] != expected:
                    raise RuntimeError("Existing path questions have different inputs")
                return len(batch)
            # Keep immutable full IDs in artifacts, but give the model short
            # batch-local references. Copying SHA-256 strings is not reasoning.
            local_to_full = {f"p{i}": p["id"] for i, p in enumerate(batch)}
            payload_paths = [{**path_prompt_item(p, graph), "id": f"p{i}"}
                             for i, p in enumerate(batch)]
            ids = set(local_to_full)
            def validate(value):
                rows = value["paths"]
                if (len(rows) != len(ids) or {p["id"] for p in rows} != ids
                        or any(not isinstance(p["questions"], list)
                               or len(p["questions"]) != 2
                               or not all(isinstance(q, str) and q.strip()
                                          for q in p["questions"]) for p in rows)):
                    raise ValueError("Path question coverage/schema mismatch")
            value, response = self.structured(
                "path_questions", digest, prompts.PATH_SYSTEM,
                {"paths": payload_paths},
                validate, max_tokens=8192,
                schema=prompts.obj({"paths": {
                    "type": "array", "minItems": len(batch), "maxItems": len(batch),
                    "items": prompts.obj({
                        "id": {"type": "string", "enum": sorted(ids)},
                        "questions": {"type": "array", "minItems": 2, "maxItems": 2,
                                      "items": prompts.STR},
                    })}}))
            for item in value["paths"]:
                item["id"] = local_to_full[item["id"]]
            write_json(path, {"input_fingerprint": expected,
                             "request_sha256": response["request_sha256"],
                             "transport_codec": "batch_local_ids_mapped_to_full_SHA256",
                             "paths": value["paths"]})
            return len(batch)
        def done(n):
            nonlocal completed
            completed += n
            self.progress("path_questions", completed=completed, total=total)
        with self.phase("path_questions"):
            parallel_work(batches(iter_paths(graph), PATH_BATCH), one,
                          workers=self.workers, on_done=done)
            repository = self.root / "path_repository.jsonl"
            written = 0
            with repository.open("w", encoding="utf-8") as stream:
                for batch in batches(iter_paths(graph), PATH_BATCH):
                    digest = signature([p["id"] for p in batch])
                    item = json.loads((self.root / "path_batches" / f"{digest}.json")
                                      .read_text(encoding="utf-8"))
                    by_id = {p["id"]: p["questions"] for p in item["paths"]}
                    for path in batch:
                        stream.write(canonical({**path, "questions": by_id[path["id"]]}) + "\n")
                        written += 1
            if written != total:
                raise RuntimeError("Incomplete path-question repository")
            write_json(self.root / "path_repository_manifest.json", {
                "paths_expected": total, "paths_completed": written,
                "questions": written * 2, "sha256": sha256(repository),
                "batch_size": PATH_BATCH, "caps": None})

    def build_indexes(self):
        graph = self.load_graph()
        with self.phase("index"):
            hyperedges = list(graph["hyperedges"].values())
            self.indexes["hyperedges"] = VectorIndex.build(
                self.root / "indexes/hyperedges",
                [{"id": p["id"]} for p in hyperedges],
                [p["title"] + "\n" + p["text"] for p in hyperedges],
                self.embeddings(), stage="hyperedge_index")
            entities = [graph["entities"][eid] for eid in graph["retained_entity_ids"]]
            # Match the supplied EntityIndexManager text_properties=['id']:
            # embed the canonical entity name, without a question-ID suffix.
            # Preserve ALL extraction descriptions and aliases in graph.json.
            self.indexes["entities"] = VectorIndex.build(
                self.root / "indexes/entities",
                [{"id": p["id"]} for p in entities],
                [p["name"] for p in entities],
                self.embeddings(), stage="entity_index")
            paths = read_jsonl(self.root / "path_repository.jsonl")
            items, texts = [], []
            for path in paths:
                for i, question in enumerate(path["questions"]):
                    items.append({"id": path["id"] + "_" + str(i), "path_id": path["id"]})
                    texts.append(question)
            self.indexes["paths"] = VectorIndex.build(
                self.root / "indexes/paths", items, texts,
                self.embeddings(), stage="path_question_index")
            write_json(self.root / "index_manifest.json", {
                "hyperedges": len(hyperedges), "entities": len(entities),
                "path_questions": len(items), "model_digest": self.embeddings().digest,
                "search": "exact_float32_cosine_no_question_id_filter",
                "entity_text": "canonical_name_as_in_supplied_entity_indexer",
                "entity_descriptions": "all_preserved_in_graph",
            })

    def load_indexes(self):
        if not self.indexes:
            for name in ("hyperedges", "entities", "paths"):
                meta = json.loads((self.root / f"indexes/{name}.json").read_text(encoding="utf-8"))
                self.indexes[name] = VectorIndex(meta["items"], np.load(
                    self.root / f"indexes/{name}.npy", mmap_mode="r"))

    def predict(self):
        graph = self.load_graph()
        self.load_indexes()
        paths = {p["id"]: p for p in read_jsonl(self.root / "path_repository.jsonl")}
        by_entity = defaultdict(list)
        for tid, triple in graph["triples"].items():
            by_entity[triple["head"]].append(tid)
            if triple["tail"] != triple["head"]:
                by_entity[triple["tail"]].append(tid)
        questions = json.loads((self.root / "inputs/questions.json").read_text(encoding="utf-8"))
        expected_keys = {"id", "question"}
        if any(set(q) != expected_keys for q in questions):
            raise RuntimeError("Inference input contains unexpected annotation fields")
        completed = 0

        def one(question):
            qid, text = question["id"], question["question"]
            destination = self.root / "predictions" / f"{qid}.json"
            if destination.exists():
                existing = json.loads(destination.read_text(encoding="utf-8"))
                if existing.get("status") == "completed":
                    return qid
            started = time.perf_counter()
            trace = []
            retrieved = set()
            calls = []
            components = defaultdict(float)
            component_lock = threading.Lock()
            def measured(label, operation, *args, **kwargs):
                began = time.perf_counter()
                try:
                    return operation(*args, **kwargs)
                finally:
                    with component_lock:
                        components[label] += time.perf_counter() - began
            def search(index, *args):
                return measured("vector_search_s_sum",
                                self.indexes[index].search, *args)
            def encode(texts, **kwargs):
                return measured("embedding_s_sum", self.embeddings().encode,
                                texts, **kwargs)
            def generate(stage, system, payload, validator, tokens=2048):
                wire = json.loads(canonical(payload))
                source_items = wire.get("passages", wire.get("background", []))
                local_to_full = {f"s{i}": p["id"] for i, p in enumerate(source_items)}
                full_to_local = {full: local for local, full in local_to_full.items()}
                for item in source_items:
                    item["id"] = full_to_local[item["id"]]
                for triple in wire.get("triples", []):
                    triple["passage_ids"] = [full_to_local[pid]
                                             for pid in triple["passage_ids"]
                                             if pid in full_to_local]
                for i, path in enumerate(wire.get("paths", [])):
                    path["id"] = f"p{i}"
                schema = json.loads(canonical(prompts.SCHEMAS[stage]))
                if "supporting_passage_ids" in schema.get("properties", {}):
                    schema["properties"]["supporting_passage_ids"]["items"] = (
                        {"type": "string", "enum": sorted(local_to_full)}
                        if local_to_full else prompts.STR)
                def wire_validator(item):
                    converted = dict(item)
                    if "supporting_passage_ids" in item:
                        if any(pid not in local_to_full for pid in item["supporting_passage_ids"]):
                            raise ValueError("Unknown local supporting source reference")
                        converted["supporting_passage_ids"] = [
                            local_to_full[pid] for pid in item["supporting_passage_ids"]]
                    validator(converted)
                value, response = measured(
                    "llm_stage_s_sum", self.structured, stage, qid, system, wire,
                    wire_validator, max_tokens=tokens, schema=schema)
                if "supporting_passage_ids" in value:
                    value["supporting_passage_ids"] = [
                        local_to_full[pid] for pid in value["supporting_passage_ids"]]
                calls.append({key: response[key] for key in
                              ("request_sha256", "stage", "elapsed_s", "cache_hit", "usage")})
                return value
            def passages(ids):
                # Keep complete original hyperedges, without sentence truncation.
                retrieved.update(ids)
                return [{"id": pid, "title": graph["hyperedges"][pid]["title"],
                         "text": graph["hyperedges"][pid]["text"]} for pid in sorted(ids)]
            def answer_schema(value):
                if (not isinstance(value, dict) or not isinstance(value.get("answer"), str)
                        or not isinstance(value.get("supporting_passage_ids"), list)):
                    raise ValueError("Invalid answer schema")
            def validate_support(value, allowed):
                answer_schema(value)
                if any(not isinstance(pid, str) or pid not in allowed
                       for pid in value["supporting_passage_ids"]):
                    raise ValueError("Supporting passage ID not in supplied evidence")
            def triple_records(tids):
                return [{
                    "head": graph["entities"][graph["triples"][tid]["head"]]["name"],
                    "relation": graph["triples"][tid]["type"],
                    "tail": graph["entities"][graph["triples"][tid]["tail"]]["name"],
                    "descriptions": graph["triples"][tid]["descriptions"],
                    "passage_ids": graph["triples"][tid]["passage_ids"],
                } for tid in sorted(tids)]

            query_vector = encode([text], stage="question_query")[0]
            # Retrieve the top THREE DISTINCT paths, using max question score.
            # Searching six questions suffices because each path has exactly two.
            question_hits = search("paths", query_vector, 6)
            hits = []
            seen = set()
            for hit in question_hits:
                if hit["path_id"] not in seen:
                    hits.append(hit)
                    seen.add(hit["path_id"])
                    if len(hits) == 3:
                        break
            direct_sources = {pid for hit in hits
                              for pid in paths[hit["path_id"]]["passage_ids"]}
            direct_sources = {hit["id"] for hit in
                              search("hyperedges", query_vector, 3, direct_sources)}
            direct_payload = {"question": text,
                              "paths": [path_prompt_item(paths[h["path_id"]], graph)
                                        for h in hits],
                              "passages": passages(direct_sources)}
            def direct_schema(value):
                validate_support(value, direct_sources)
                if not isinstance(value.get("sufficient"), bool):
                    raise ValueError("Invalid evidence-sufficiency field")
                if value["sufficient"] and (not value["answer"].strip()
                                            or not value["supporting_passage_ids"]):
                    raise ValueError("Sufficient answer has no answer/evidence")
            if hits:
                direct = generate("direct", prompts.DIRECT_SYSTEM, direct_payload, direct_schema)
            else:
                direct = {"sufficient": False, "answer": "", "supporting_passage_ids": []}
            trace.append({"stage": "direct", "path_hits": hits, "result": direct})
            if direct["sufficient"]:
                final = direct
                route = "direct"
            else:
                background = search("hyperedges", query_vector, 2)
                background_ids = {p["id"] for p in background}
                def decomp_schema(value):
                    subs = value["subquestions"]
                    if not isinstance(subs, list) or not 1 <= len(subs) <= 6:
                        raise ValueError("Invalid decomposition length")
                    for sub in subs:
                        if (not isinstance(sub.get("question"), str)
                                or not sub["question"].strip()
                                or not isinstance(sub.get("entities"), list)
                                or not all(isinstance(e, str) for e in sub["entities"])
                                or not isinstance(sub.get("needs_external"), bool)):
                            raise ValueError("Invalid subquestion schema")
                decomposition = generate(
                    "decomposition", prompts.DECOMPOSE_SYSTEM,
                    {"question": text, "background": passages(background_ids)}, decomp_schema)
                trace.append({"stage": "decomposition", "result": decomposition})
                history = []
                final_sources = set(background_ids)
                final_triples = set()
                def answer_subquestion(sub, prior_history):
                    current = dict(sub)
                    if current["needs_external"] and prior_history:
                        def rewrite_schema(value):
                            if (not isinstance(value.get("question"), str)
                                    or not value["question"].strip()
                                    or not isinstance(value.get("entities"), list)
                                    or not all(isinstance(e, str) for e in value["entities"])
                                    or not isinstance(value.get("needs_external"), bool)):
                                raise ValueError("Invalid rewrite schema")
                        current = generate(
                            "rewrite", prompts.REWRITE_SYSTEM,
                            {"original_question": text, "history": prior_history,
                             "current_subquestion": current}, rewrite_schema)
                    vector = encode([current["question"]], stage="subquestion_query")[0]
                    names = current["entities"]
                    linking_vectors = (encode(
                        names, stage="entity_linking_query") if names else [])
                    by_id = {}
                    for linking_vector in linking_vectors:
                        for hit in search("entities", linking_vector, 2):
                            if hit["score"] > by_id.get(hit["id"], {}).get("score", -2):
                                by_id[hit["id"]] = hit
                    entity_hits = sorted(by_id.values(),
                                         key=lambda h: (-h["score"], h["id"]))[:2]
                    # A fixed 0.3 cosine threshold is inherited from settings.py;
                    # it is a model-space match heuristic, never tuned on gold.
                    entity_hits = [e for e in entity_hits if e["score"] >= 0.3]
                    tids = {tid for hit in entity_hits for tid in by_entity[hit["id"]]}
                    associated = {pid for tid in tids
                                  for pid in graph["triples"][tid]["passage_ids"]}
                    associated.update(pid for hit in entity_hits
                                      for pid in graph["entities"][hit["id"]]["passage_ids"])
                    if associated:
                        passage_hits = search("hyperedges", vector, 3, associated)
                        retrieval_route = "entity_and_hyperedge"
                    else:
                        passage_hits = search("hyperedges", vector, 2)
                        retrieval_route = "hyperedge_fallback"
                    source_ids = {hit["id"] for hit in passage_hits}
                    evidence = passages(source_ids)
                    value = generate(
                        "subquestion", prompts.STEP_SYSTEM,
                        {"subquestion": current["question"],
                         "triples": triple_records(tids), "passages": evidence},
                        lambda value: validate_support(value, source_ids))
                    return ({"question": current["question"],
                             "answer": value["answer"],
                             "supporting_passage_ids": value["supporting_passage_ids"]},
                            {"stage": "subquestion", "original": sub,
                             "rewritten": current, "entity_hits": entity_hits,
                             "passage_hits": passage_hits, "triple_ids": sorted(tids),
                             "retrieval_route": retrieval_route, "result": value},
                            source_ids, tids)

                subquestions = decomposition["subquestions"]
                position = 0
                while position < len(subquestions):
                    stop = position + 1
                    if not subquestions[position]["needs_external"]:
                        while (stop < len(subquestions)
                               and not subquestions[stop]["needs_external"]):
                            stop += 1
                    wave = subquestions[position:stop]
                    prior_history = list(history)
                    # Independent questions in a wave use no results from one
                    # another. The client's shared semaphore still limits ALL
                    # calls across questions and waves to self.workers.
                    if len(wave) > 1 and self.workers > 1:
                        with ThreadPoolExecutor(max_workers=min(
                                self.workers, len(wave))) as pool:
                            futures = [pool.submit(answer_subquestion, sub, prior_history)
                                       for sub in wave]
                            results = [future.result() for future in futures]
                    else:
                        results = [answer_subquestion(sub, prior_history) for sub in wave]
                    for history_item, trace_item, source_ids, tids in results:
                        history.append(history_item)
                        trace.append(trace_item)
                        final_sources.update(source_ids)
                        final_triples.update(tids)
                    position = stop
                final = generate(
                    "final", prompts.FINAL_SYSTEM,
                    {"question": text, "history": history,
                     "triples": triple_records(final_triples),
                     "passages": passages(final_sources)},
                    lambda value: validate_support(value, final_sources))
                route = "progressive"
            result = {
                "id": qid, "question": text, "status": "completed",
                "answer": final["answer"].strip(),
                "supporting_passage_ids": final["supporting_passage_ids"],
                "retrieved_passage_ids": sorted(retrieved), "route": route,
                "elapsed_s": time.perf_counter() - started,
                "component_seconds": dict(components),
                "component_time_note": (
                    "Service-time sums may overlap for parallel subquestions. "
                    "Embedding time includes its shared local-service lock; LLM "
                    "stage time includes semaphore wait, retries and validation."),
                "model_calls": calls, "trace": trace,
                "model": MODEL, "reasoning_effort": EFFORT,
            }
            write_json(destination, result)
            return qid
        def done(qid):
            nonlocal completed
            completed += 1
            self.progress("prediction", completed=completed, total=len(questions), last_id=qid)
        def tracked_one(question):
            qid = question["id"]
            destination = self.root / "predictions" / (qid + ".json")
            if destination.exists():
                saved = json.loads(destination.read_text(encoding="utf-8"))
                if saved.get("status") == "completed":
                    return one(question)
            began = time.perf_counter()
            attempt = {"id": qid, "pid": os.getpid(), "started_unix": time.time()}
            append_jsonl(self.root / "prediction_attempts.jsonl",
                         {**attempt, "status": "started"})
            try:
                result = one(question)
            except Exception as error:
                append_jsonl(self.root / "prediction_attempts.jsonl", {
                    **attempt, "status": "failed", "ended_unix": time.time(),
                    "elapsed_s": time.perf_counter() - began,
                    "error_type": type(error).__name__,
                    "error_category": getattr(error, "category", None)})
                raise
            append_jsonl(self.root / "prediction_attempts.jsonl", {
                **attempt, "status": "completed", "ended_unix": time.time(),
                "elapsed_s": time.perf_counter() - began})
            return result
        with self.phase("prediction"):
            parallel_work(questions, tracked_one, workers=self.workers, on_done=done)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--source", type=Path, default=ROOT.with_name("SHGraphRAG"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=(
        "prepare", "extract", "graph", "paths", "index", "predict", "evaluate", "all"),
        default="all")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if not args.output.is_relative_to(ROOT):
        raise ValueError("All reproduction outputs must be inside this worktree")
    if not 1 <= args.workers <= 64:
        raise ValueError("OpenRouter workload concurrency must be in 1..64")
    args.output.mkdir(parents=True, exist_ok=True)
    config = {
        "dataset": args.dataset, "limit": args.limit, "workers": args.workers,
        "method_version": METHOD_VERSION,
        "source": str(args.source.resolve()), "model": MODEL, "reasoning_effort": EFFORT,
        "provider": PROVIDER, "base_url": BASE_URL, "api": "chat_completions",
        "temperature": 0,
        "reasoning_configuration": "non_reasoning_model_no_parameter",
        "transport_timeout_s": 240, "transport_max_attempts": 5,
        "retry_incomplete_chunked_response": True,
        "prediction_attempt_telemetry": "started_terminal_events_v1",
        "retryable_finish_reasons": ["content_filter", "error"],
        "malformed_finish_reason_retry": True,
        "atomic_json_replace_max_attempts": 8,
        "structured_max_attempts": 3, "resource_sample_interval_s": 5,
        "endpoint_completion": "fixed_missing_endpoint_records_v1_after_three_invalid_outputs",
        "endpoint_completion_max_attempts": 3,
        "max_subquestions": 6, "path_length_edges": 2,
        "questions_per_path": 2, "embedding_model": "bge-m3:latest",
        "embedding_dimensions": 1024, "embedding_batch_size": 32,
        "path_embedding_cache": "model_digest_exact_text_all_rows_retained",
        "embedding_keep_alive": "5m",
        "embedding_truncate": False, "vector_dtype": "float32",
        "vector_search": "exact_cosine_stable_tie_breaking",
        "seed": 20261020, "llm_seed": None,
        "llm_seed_note": "No deterministic seed support established for this endpoint",
        "top_paths": 3, "top_entities": 2, "top_hyperedges": 3,
        "fallback_hyperedges": 2, "entity_cosine_threshold": 0.3,
        "entity_merge_threshold": 0.95, "entity_merge_batch_size": 10,
        "entity_merge_scope": "unlabelled_candidate_groups_construction_only",
        "entity_index_text": "canonical_name",
        "independent_subquestions": "parallel_waves_shared_request_semaphore",
        "path_batch_size": PATH_BATCH, "path_caps": None,
        "storage": "isolated_JSON_graph_SQLite_embedding_cache_exact_cosine_numpy",
        "retrieval_scope": "global_candidate_passage_union",
    }
    config_path = args.output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("Cannot reuse run directory with a different configuration")
    write_json(config_path, config)
    files = {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")}
    code_path = args.output / "code_manifest.json"
    if code_path.exists():
        previous = json.loads(code_path.read_text(encoding="utf-8"))
        if previous.get("modules_sha256") != files:
            append_jsonl(args.output / "code_manifest_history.jsonl", {
                "recorded_unix": time.time(), "previous_manifest": previous,
                "reason": "Explicit implementation transition; previous generated artifacts retained."})
    write_json(code_path, {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "modules_sha256": files,
        "supporting_files_sha256": {
            name: sha256(ROOT / name) for name in (
                "config/prompt.py", "reproduction/requirements.lock.txt",
                "reproduction/METHOD_FIDELITY.md")},
    })
    stages = (["prepare", "extract", "graph", "paths", "index", "predict", "evaluate"]
              if args.stage == "all" else [args.stage])
    runner = None
    with RunTelemetry(args.output):
        for stage in stages:
            if stage == "prepare":
                started = time.perf_counter()
                print(canonical(prepare(args.source, args.output, args.dataset, args.limit)), flush=True)
                append_jsonl(args.output / "timing.jsonl", {
                    "stage": "prepare", "elapsed_s": time.perf_counter() - started})
            elif stage == "evaluate":
                started = time.perf_counter()
                print(canonical(evaluate_run(args.output)), flush=True)
                append_jsonl(args.output / "timing.jsonl", {
                    "stage": "evaluation", "elapsed_s": time.perf_counter() - started})
            else:
                if runner is None:
                    runner = Runner(args.output, args.workers)
                getattr(runner, {"extract": "extract", "graph": "build_graph",
                                 "paths": "path_questions", "index": "build_indexes",
                                 "predict": "predict"}[stage])()


if __name__ == "__main__":
    main()
