"""Reconstruct every completed QA request from retained inference evidence only."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from reproduction import prompts
from reproduction.client import (
    RUNTIME_DIRECTORY, canonical, parse_json, request_body, signature, write_json)
from reproduction.data import sha256
from reproduction.embedding import VectorIndex
from reproduction.graph import path_prompt_item
from reproduction.helpers.audit_response_artifacts import load, require, response_for
from reproduction.helpers.audit_index_provenance import DIGEST
from reproduction.helpers.storage_paths import checked_run_path


class QAAudit:
    def __init__(self, run):
        self.run = run
        self.graph = load(run / "graph.json")
        self.predictions = {
            p.stem: load(p) for p in (run / "predictions").glob("*.json")}
        needed_paths = {
            hit["path_id"] for prediction in self.predictions.values()
            for hit in prediction["trace"][0]["path_hits"]}
        self.paths = {}
        with (run / "path_repository.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                path = json.loads(line)
                if path["id"] in needed_paths:
                    self.paths[path["id"]] = path
        require(set(self.paths) == needed_paths, "Saved QA references unknown paths")
        metadata = load(run / "indexes/hyperedges.json")
        index_audit = load(run / "index_provenance_audit.json")
        accepted = next(r for r in index_audit["indexes"] if r["name"] == "hyperedges")
        require(
            accepted["metadata_sha256"] == sha256(run / "indexes/hyperedges.json")
            and accepted["matrix_sha256"] == sha256(run / "indexes/hyperedges.npy"),
            "Previously accepted hyperedge index bytes differ")
        self.hyperedges = VectorIndex(
            metadata["items"], np.load(run / "indexes/hyperedges.npy", mmap_mode="r"))
        self.by_entity = defaultdict(set)
        for tid, triple in self.graph["triples"].items():
            self.by_entity[triple["head"]].add(tid)
            self.by_entity[triple["tail"]].add(tid)
        cache = ROOT / RUNTIME_DIRECTORY / "embedding_cache.sqlite3"
        self.connection = sqlite3.connect(cache.resolve().as_uri() + "?mode=ro", uri=True)
        self.vector_proofs = {}

    def vector(self, text):
        key = signature({"model_digest": DIGEST, "text": text})
        row = self.connection.execute(
            "SELECT vector FROM embeddings WHERE key=?", (key,)).fetchone()
        require(row is not None and len(row[0]) == 4096, "Missing retained query vector")
        vector = np.frombuffer(row[0], dtype=np.float32)
        require(np.isfinite(vector).all()
                and np.isclose(np.linalg.norm(vector), 1, atol=1e-4),
                "Invalid retained query vector")
        self.vector_proofs[key] = hashlib.sha256(row[0]).hexdigest()
        return vector

    def passages(self, ids):
        self.retrieved.update(ids)
        return [{"id": pid, "title": self.graph["hyperedges"][pid]["title"],
                 "text": self.graph["hyperedges"][pid]["text"]} for pid in sorted(ids)]

    def triples(self, ids):
        graph = self.graph
        return [{
            "head": graph["entities"][graph["triples"][tid]["head"]]["name"],
            "relation": graph["triples"][tid]["type"],
            "tail": graph["entities"][graph["triples"][tid]["tail"]]["name"],
            "descriptions": graph["triples"][tid]["descriptions"],
            "passage_ids": graph["triples"][tid]["passage_ids"],
        } for tid in sorted(ids)]

    def generate(self, stage, system, payload, validator):
        wire = json.loads(canonical(payload))
        sources = wire.get("passages", wire.get("background", []))
        mapping = {f"s{i}": p["id"] for i, p in enumerate(sources)}
        reverse = {full: local for local, full in mapping.items()}
        for item in sources:
            item["id"] = reverse[item["id"]]
        for triple in wire.get("triples", []):
            triple["passage_ids"] = [
                reverse[pid] for pid in triple["passage_ids"] if pid in reverse]
        for index, path in enumerate(wire.get("paths", [])):
            path["id"] = f"p{index}"
        schema = json.loads(canonical(prompts.SCHEMAS[stage]))
        if "supporting_passage_ids" in schema.get("properties", {}):
            schema["properties"]["supporting_passage_ids"]["items"] = (
                {"type": "string", "enum": sorted(mapping)} if mapping else prompts.STR)
        prompt, proofs = canonical(wire), []
        for repair in range(3):
            response, proof = response_for(request_body(system, prompt, 2048, schema))
            proofs.append(proof)
            try:
                value = parse_json(response["text"])
                converted = dict(value)
                if "supporting_passage_ids" in value:
                    if any(pid not in mapping for pid in value["supporting_passage_ids"]):
                        raise ValueError("Unknown local supporting source reference")
                    converted["supporting_passage_ids"] = [
                        mapping[pid] for pid in value["supporting_passage_ids"]]
                validator(converted)
                break
            except (ValueError, TypeError, KeyError, AssertionError) as error:
                if repair == 2:
                    raise ValueError("Saved QA structural repair is invalid") from error
                prompt = (
                    canonical(wire) + "\nThe previous output did not match "
                    "the required complete JSON schema. Return a full corrected "
                    "replacement, with no extra text. Error: " + str(error)
                    + "\nPrevious output:\n" + response["text"])
        recorded = self.calls[response["request_sha256"]]
        require(bool(recorded), "Reconstructed QA request absent from saved model calls")
        call = recorded.pop()
        require(call["usage"] == response["usage"] and call["stage"] == response["stage"],
                "Saved QA call usage/stage differs from retained response")
        self.proofs.append({
            "stage": stage, "schema_repairs": repair, "responses": proofs,
            "final_response_text_sha256": hashlib.sha256(
                response["text"].encode("utf-8")).hexdigest(),
            "recorded_cache_hit": call["cache_hit"],
        })
        return converted

    @staticmethod
    def support(value, allowed):
        if (not isinstance(value, dict) or not isinstance(value.get("answer"), str)
                or not isinstance(value.get("supporting_passage_ids"), list)):
            raise ValueError("Invalid answer schema")
        if any(not isinstance(pid, str) or pid not in allowed
               for pid in value["supporting_passage_ids"]):
            raise ValueError("Supporting passage ID not in supplied evidence")

    @staticmethod
    def decomposition(value):
        subs = value["subquestions"]
        if not isinstance(subs, list) or not 1 <= len(subs) <= 6:
            raise ValueError("Invalid decomposition length")
        for sub in subs:
            if (not isinstance(sub.get("question"), str) or not sub["question"].strip()
                    or not isinstance(sub.get("entities"), list)
                    or not all(isinstance(e, str) for e in sub["entities"])
                    or not isinstance(sub.get("needs_external"), bool)):
                raise ValueError("Invalid subquestion schema")

    @staticmethod
    def rewrite(value):
        if (not isinstance(value.get("question"), str) or not value["question"].strip()
                or not isinstance(value.get("entities"), list)
                or not all(isinstance(e, str) for e in value["entities"])
                or not isinstance(value.get("needs_external"), bool)):
            raise ValueError("Invalid rewrite schema")

    def question(self, item):
        prediction = self.predictions[item["id"]]
        require(prediction["question"] == item["question"]
                and prediction["status"] == "completed", "QA input identity differs")
        self.retrieved, self.calls, self.proofs = set(), defaultdict(list), []
        for call in prediction["model_calls"]:
            self.calls[call["request_sha256"]].append(call)
        trace = prediction["trace"]
        require(trace[0]["stage"] == "direct", "Missing direct trace")
        hits = trace[0]["path_hits"]
        require(len(hits) <= 3 and len({h["path_id"] for h in hits}) == len(hits),
                "Direct path count/uniqueness differs")
        query = self.vector(item["question"])
        direct_sources = {pid for hit in hits for pid in self.paths[hit["path_id"]]["passage_ids"]}
        direct_sources = {h["id"] for h in self.hyperedges.search(query, 3, direct_sources)}

        def direct_schema(value):
            self.support(value, direct_sources)
            if not isinstance(value.get("sufficient"), bool):
                raise ValueError("Invalid evidence-sufficiency field")
            if value["sufficient"] and (not value["answer"].strip()
                                       or not value["supporting_passage_ids"]):
                raise ValueError("Sufficient answer has no answer/evidence")

        direct = (self.generate(
            "direct", prompts.DIRECT_SYSTEM, {
                "question": item["question"],
                "paths": [path_prompt_item(self.paths[h["path_id"]], self.graph) for h in hits],
                "passages": self.passages(direct_sources)}, direct_schema)
            if hits else {"sufficient": False, "answer": "", "supporting_passage_ids": []})
        require(direct == trace[0]["result"], "Direct response differs from saved trace")
        if direct["sufficient"]:
            require(prediction["route"] == "direct" and len(trace) == 1, "Direct route differs")
            final = direct
        else:
            require(prediction["route"] == "progressive"
                    and len(trace) >= 2 and trace[1]["stage"] == "decomposition",
                    "Progressive route/trace differs")
            background = {h["id"] for h in self.hyperedges.search(query, 2)}
            decomp = self.generate(
                "decomposition", prompts.DECOMPOSE_SYSTEM,
                {"question": item["question"], "background": self.passages(background)},
                self.decomposition)
            require(decomp == trace[1]["result"], "Decomposition response differs")
            subs = decomp["subquestions"]
            require(len(trace) == len(subs) + 2, "Subquestion trace coverage differs")
            history, final_sources, final_triples = [], set(background), set()
            position = 0
            while position < len(subs):
                stop = position + 1
                if not subs[position]["needs_external"]:
                    while stop < len(subs) and not subs[stop]["needs_external"]:
                        stop += 1
                prior = list(history)
                for index in range(position, stop):
                    sub, saved = subs[index], trace[index + 2]
                    require(saved["stage"] == "subquestion" and saved["original"] == sub,
                            "Subquestion identity differs")
                    current = dict(sub)
                    if current["needs_external"] and prior:
                        current = self.generate(
                            "rewrite", prompts.REWRITE_SYSTEM, {
                                "original_question": item["question"], "history": prior,
                                "current_subquestion": current}, self.rewrite)
                    require(current == saved["rewritten"], "Rewrite response differs")
                    entity_hits = saved["entity_hits"]
                    require(len(entity_hits) <= 2
                            and all(h["id"] in self.graph["retained_entity_ids"]
                                    and h["score"] >= 0.3 for h in entity_hits),
                            "Saved entity selection boundary differs")
                    tids = {tid for h in entity_hits for tid in self.by_entity[h["id"]]}
                    require(sorted(tids) == saved["triple_ids"], "Incident triple evidence differs")
                    associated = {pid for tid in tids
                                  for pid in self.graph["triples"][tid]["passage_ids"]}
                    associated.update(pid for h in entity_hits
                                      for pid in self.graph["entities"][h["id"]]["passage_ids"])
                    vector = self.vector(current["question"])
                    passage_hits = self.hyperedges.search(
                        vector, 3, associated) if associated else self.hyperedges.search(vector, 2)
                    require([h["id"] for h in passage_hits]
                            == [h["id"] for h in saved["passage_hits"]],
                            "Hyperedge retrieval IDs/order differ")
                    expected_route = "entity_and_hyperedge" if associated else "hyperedge_fallback"
                    require(saved["retrieval_route"] == expected_route, "Fallback route differs")
                    source_ids = {h["id"] for h in passage_hits}
                    value = self.generate(
                        "subquestion", prompts.STEP_SYSTEM, {
                            "subquestion": current["question"],
                            "triples": self.triples(tids), "passages": self.passages(source_ids)},
                        lambda result: self.support(result, source_ids))
                    require(value == saved["result"], "Subquestion answer differs")
                    history.append({"question": current["question"], "answer": value["answer"],
                                    "supporting_passage_ids": value["supporting_passage_ids"]})
                    final_sources.update(source_ids)
                    final_triples.update(tids)
                position = stop
            final = self.generate(
                "final", prompts.FINAL_SYSTEM, {
                    "question": item["question"], "history": history,
                    "triples": self.triples(final_triples),
                    "passages": self.passages(final_sources)},
                lambda value: self.support(value, final_sources))
        require(final["answer"].strip() == prediction["answer"]
                and final["supporting_passage_ids"] == prediction["supporting_passage_ids"],
                "Final answer/support differs from reconstructed response")
        require(sorted(self.retrieved) == prediction["retrieved_passage_ids"],
                "Retrieved source union differs")
        require(not any(self.calls.values()), "Unmatched recorded QA model calls")
        return {
            "id": item["id"], "artifact_sha256": sha256(
                self.run / "predictions" / (item["id"] + ".json")),
            "route": prediction["route"], "model_calls_verified": len(self.proofs),
            "calls": self.proofs,
        }


def audit(run):
    began = time.perf_counter()
    code = load(run / "code_manifest.json")
    for name, expected in code["modules_sha256"].items():
        require(sha256(ROOT / "reproduction" / name) == expected, "Frozen research source differs")
    manifest = load(run / "data_manifest.json")
    for name, expected in manifest["inputs_sha256"].items():
        require(sha256(run / "inputs" / name) == expected, "Frozen inference input differs")
    questions = load(run / "inputs/questions.json")
    ids = load(run / "inputs/ids.json")
    require([q["id"] for q in questions] == ids
            and all(set(q) == {"id", "question"} for q in questions), "Label-free IDs differ")
    api_hash = sha256(run / "api_calls.jsonl")
    evidence = QAAudit(run)
    try:
        require(set(evidence.predictions) == set(ids), "QA coverage differs")
        results = []
        for index, question in enumerate(questions):
            results.append(evidence.question(question))
            if (index + 1) % 100 == 0:
                print(json.dumps({"qa_responses_verified": index + 1, "expected": len(ids)}),
                      flush=True)
        require(sha256(run / "api_calls.jsonl") == api_hash, "Remote log changed during audit")
        result = {
            "status": "passed", "observed_unix": time.time(),
            "elapsed_s": time.perf_counter() - began, "questions": len(results),
            "complete_prediction_coverage": len(results) == manifest["questions"],
            "model_calls_verified": sum(r["model_calls_verified"] for r in results),
            "route_counts": dict(Counter(r["route"] for r in results)),
            "structural_corrections_verified": sum(
                c["schema_repairs"] for r in results for c in r["calls"]),
            "new_model_calls": 0, "new_embedding_calls": 0, "gold_read": False,
            "helper_sha256": sha256(Path(__file__)), "core_hashes": code["modules_sha256"],
            "remote_api_log_sha256_unchanged": api_hash,
            "retained_query_vectors_verified": evidence.vector_proofs,
            "questions_evidence": results,
            "scope": "Complete QA request/response reconstruction given retained path/entity "
                     "selections. Hyperedge evidence selection, wave/dependency history, "
                     "schema repairs, local/full source references and final answers checked.",
            "limitations": [
                "Path/entity nearest-neighbor rankings are not replayed; saved selections "
                "are inputs to this response-provenance audit.",
                "Response artifacts and raw cache are local evidence, not independent "
                "remote billing records or factual-answer validation.",
                "Audit time is separate from model construction/query measurement.",
            ],
        }
    finally:
        evidence.connection.close()
    write_json(run / "qa_response_provenance_audit.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    result = audit(checked_run_path(args.run))
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("questions_evidence", "retained_query_vectors_verified",
                                   "core_hashes")}, indent=2))
