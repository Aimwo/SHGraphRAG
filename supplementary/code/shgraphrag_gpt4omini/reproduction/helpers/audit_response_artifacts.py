"""Audit saved artifacts against exact cached OpenRouter requests without API calls.

Use --require-complete for final stage acceptance. The default audits the
immutable files already saved at the start, with partial coverage explicit.
This is transport and artifact provenance, not factual-quality evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from reproduction import prompts
from reproduction.client import (EFFORT, MODEL, PROVIDER, RUNTIME_DIRECTORY,
                                 canonical, parse_json, signature, write_json)
from reproduction.data import read_jsonl, sha256
from reproduction.graph import entity_key, iter_paths, parse_json_extraction, path_prompt_item

CACHE = ROOT / RUNTIME_DIRECTORY / "chat_completions"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def request_body(system, prompt, schema):
    return {
        "model": MODEL, "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}],
        "max_tokens": 8192, "temperature": 0, "stream": False,
        "provider": {"require_parameters": True},
        "response_format": {
            "type": "json_schema", "json_schema": {
                "name": "research_output", "strict": True, "schema": schema}},
    }


def response_for(body):
    fingerprint = signature(body)
    path = CACHE / fingerprint[:2] / (fingerprint + ".json")
    response = load(path)
    require(response.get("request_sha256") == fingerprint, "Cache request hash mismatch")
    require(response.get("model") == MODEL and response.get("provider") == PROVIDER
            and response.get("reasoning_effort") == EFFORT
            and response.get("status") == "completed", "Cache model/effort/status mismatch")
    reasoning = response.get("usage", {}).get("output_tokens_details", {})
    require(reasoning.get("reasoning_tokens", 0) == 0, "Unexpected reasoning usage")
    return response, {
        "request_sha256": fingerprint, "response_cache_sha256": sha256(path),
        "response_id": response.get("response_id"),
    }


def audit_extractions(run, corpus, *, require_complete):
    saved = {p.stem: p for p in (run / "extractions").glob("*.json")}
    expected_ids = {p["passage_id"] for p in corpus}
    require(not set(saved) - expected_ids, "Unknown extraction IDs")
    require(not require_complete or set(saved) == expected_ids, "Extraction coverage incomplete")
    evidence = []
    for passage in corpus:
        pid = passage["passage_id"]
        if pid not in saved:
            continue
        artifact = load(saved[pid])
        prompt = prompts.extraction_input(passage)
        require(artifact.get("passage_id") == pid, "Extraction passage ID mismatch")
        require(artifact.get("input_fingerprint") == signature(
            [prompts.EXTRACT_SYSTEM, prompt]), "Extraction input fingerprint mismatch")
        responses = []
        for repair in range(3):
            response, proof = response_for(request_body(
                prompts.EXTRACT_SYSTEM, prompt, prompts.EXTRACT_SCHEMA))
            responses.append(proof)
            try:
                parsed = parse_json_extraction(parse_json(response["text"]))
                break
            except (ValueError, TypeError, KeyError) as error:
                if repair == 2:
                    if artifact.get("endpoint_completion") and str(error).startswith(
                            "Relationship endpoint absent from entity records:"):
                        parsed, response, closure_proofs = audit_endpoint_completion(
                            passage, response, artifact["endpoint_completion"])
                        responses.extend(closure_proofs)
                        break
                    raise ValueError("Cached extraction repair remains invalid") from error
                prompt = (
                    prompts.extraction_input(passage)
                    + "\nRepair the previous output to satisfy ALL JSON "
                      "schema rules. Every relationship endpoint must "
                      "have an entity record. Add complete entity records "
                      "for EVERY missing endpoint named in the error. "
                      "Keep all valid records. Return the FULL corrected "
                      "JSON object. Error: " + str(error)
                    + "\nPrevious output:\n" + response["text"])
        require(artifact.get("request_sha256") == responses[-1]["request_sha256"],
                "Extraction request does not match reconstructed request")
        require(artifact.get("extracted") == parsed, "Extraction differs from cached response")
        evidence.append({
            "passage_id": pid, "artifact_sha256": sha256(saved[pid]),
            "repair_count": len(responses) - 1, "responses": responses,
        })
    return {
        "stage": "extraction", "expected_artifacts": len(corpus),
        "verified_artifacts": len(evidence), "complete_stage_coverage": len(saved) == len(corpus),
        "schema_repair_count": sum(e["repair_count"] for e in evidence),
        "artifacts": evidence,
    }


def audit_endpoint_completion(passage, base_response, evidence):
    require(evidence.get("codec") == "fixed_missing_endpoint_records_v1",
            "Unknown endpoint completion codec")
    require(evidence.get("base_request_sha256") == base_response["request_sha256"],
            "Endpoint completion base request mismatch")
    original = parse_json(base_response["text"])
    known = {entity_key(e["name"]) for e in original["entities"]}
    names = sorted({r[field] for r in original["relationships"]
                    for field in ("source", "target") if entity_key(r[field]) not in known})
    require(names == evidence["required_names"], "Endpoint completion names differ")
    schema = prompts.obj({"records": prompts.obj({
        name: prompts.obj({"type": prompts.STR, "description": prompts.STR})
        for name in names})})
    payload = {
        "passage": {"title": passage["title"], "text": passage["text"]},
        "required_names": names, "existing_entities": original["entities"],
        "relationships": original["relationships"]}
    prompt, proofs = canonical(payload), []
    for repair in range(3):
        response, proof = response_for(request_body(prompts.ENDPOINT_SYSTEM, prompt, schema))
        proofs.append(proof)
        try:
            records = parse_json(response["text"])["records"]
            if set(records) != set(names) or any(
                    not all(isinstance(record.get(k), str) and record[k].strip()
                            for k in ("type", "description"))
                    for record in records.values()):
                raise ValueError("Incomplete fixed-name endpoint fields")
            break
        except (ValueError, TypeError, KeyError, AssertionError) as error:
            if repair == 2:
                raise
            prompt = (
                canonical(payload) + "\nThe previous output did not match "
                "the required complete JSON schema. Return a full corrected "
                "replacement, with no extra text. Error: " + str(error)
                + "\nPrevious output:\n" + response["text"])
    require(records == evidence["records"], "Endpoint records differ from model response")
    require(response["request_sha256"] == evidence["response_request_sha256"],
            "Endpoint completion response hash mismatch")
    combined = {
        "entities": original["entities"] + [
            {"name": name, **records[name]} for name in names],
        "relationships": original["relationships"]}
    return parse_json_extraction(combined), response, proofs


def chunks(iterator, size):
    batch = []
    for item in iterator:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def validate_path_response(value, ids):
    rows = value["paths"]
    if (len(rows) != len(ids) or {p["id"] for p in rows} != ids
            or any(not isinstance(p["questions"], list)
                   or len(p["questions"]) != 2
                   or not all(isinstance(q, str) and q.strip()
                              for q in p["questions"]) for p in rows)):
        raise ValueError("Path question coverage/schema mismatch")


def audit_paths(run, *, require_complete):
    graph = load(run / "graph.json")
    saved = {p.stem: p for p in (run / "path_batches").glob("*.json")}
    evidence, observed_batches, expected_batches = [], set(), set()
    paths_seen = verified_paths = 0
    for batch in chunks(iter_paths(graph), 32):
        paths_seen += len(batch)
        digest = signature([p["id"] for p in batch])
        require(digest not in expected_batches, "Duplicate enumerated path batch")
        expected_batches.add(digest)
        if digest not in saved:
            continue
        artifact = load(saved[digest])
        expected_input = signature([
            prompts.PATH_SYSTEM, [path_prompt_item(p, graph) for p in batch]])
        require(artifact.get("input_fingerprint") == expected_input,
                "Path batch input fingerprint mismatch")
        mapping = {f"p{i}": p["id"] for i, p in enumerate(batch)}
        ids = set(mapping)
        payload = {"paths": [{**path_prompt_item(p, graph), "id": f"p{i}"}
                             for i, p in enumerate(batch)]}
        schema = prompts.obj({"paths": {
            "type": "array", "minItems": len(batch), "maxItems": len(batch),
            "items": prompts.obj({
                "id": {"type": "string", "enum": sorted(ids)},
                "questions": {"type": "array", "minItems": 2, "maxItems": 2,
                              "items": prompts.STR}})}})
        prompt, responses = canonical(payload), []
        for repair in range(3):
            response, proof = response_for(request_body(prompts.PATH_SYSTEM, prompt, schema))
            responses.append(proof)
            try:
                value = parse_json(response["text"])
                validate_path_response(value, ids)
                break
            except (ValueError, TypeError, KeyError, AssertionError) as error:
                if repair == 2:
                    raise ValueError("Cached path repair remains invalid") from error
                prompt = (
                    canonical(payload) + "\nThe previous output did not match "
                    "the required complete JSON schema. Return a full corrected "
                    "replacement, with no extra text. Error: " + str(error)
                    + "\nPrevious output:\n" + response["text"])
        for item in value["paths"]:
            item["id"] = mapping[item["id"]]
        require(artifact.get("request_sha256") == responses[-1]["request_sha256"],
                "Path batch request does not match reconstructed request")
        require(artifact.get("paths") == value["paths"], "Path batch differs from cached response")
        observed_batches.add(digest)
        verified_paths += len(batch)
        evidence.append({
            "batch_id": digest, "artifact_sha256": sha256(saved[digest]),
            "paths": len(batch), "repair_count": len(responses) - 1, "responses": responses,
        })
    require(paths_seen == graph["stats"]["two_edge_paths"], "Enumerated path count differs")
    require(not set(saved) - observed_batches, "Unknown or unverified path batches")
    complete = observed_batches == expected_batches
    require(not require_complete or complete, "Path question coverage incomplete")
    repository_verified = False
    if require_complete:
        # Compare every complete path and its two questions, streaming in the
        # deterministic construction order without loading the repository.
        count = 0
        with (run / "path_repository.jsonl").open(encoding="utf-8") as repository:
            for batch in chunks(iter_paths(graph), 32):
                digest = signature([p["id"] for p in batch])
                by_id = {p["id"]: p["questions"] for p in load(saved[digest])["paths"]}
                for path in batch:
                    line = repository.readline()
                    require(bool(line), "Path repository is truncated")
                    require(json.loads(line) == {**path, "questions": by_id[path["id"]]},
                            "Path repository differs from audited response batches")
                    count += 1
            require(not repository.read().strip(), "Path repository has extra records")
        manifest = load(run / "path_repository_manifest.json")
        require(manifest["paths_expected"] == manifest["paths_completed"] == count
                and manifest["questions"] == count * 2 and manifest["caps"] is None,
                "Path manifest coverage/cap mismatch")
        require(manifest["sha256"] == sha256(run / "path_repository.jsonl"),
                "Path repository hash mismatch")
        repository_verified = True
    return {
        "stage": "path_questions", "expected_artifacts": len(expected_batches),
        "verified_artifacts": len(evidence), "expected_paths": paths_seen,
        "verified_paths": verified_paths, "complete_stage_coverage": complete,
        "repository_verified_record_by_record": repository_verified,
        "schema_repair_count": sum(e["repair_count"] for e in evidence),
        "artifacts": evidence,
    }


def audit(run, stages, *, require_complete=False):
    config = load(run / "config.json")
    require(config["model"] == MODEL and config["reasoning_effort"] == EFFORT,
            "Run model/effort differs")
    code = load(run / "code_manifest.json")
    for name, expected in code["modules_sha256"].items():
        require(sha256(ROOT / "reproduction" / name) == expected, "Frozen module drift")
    manifest = load(run / "data_manifest.json")
    for name, expected in manifest["inputs_sha256"].items():
        require(sha256(run / "inputs" / name) == expected, "Frozen inference input drift")
    corpus = read_jsonl(run / "inputs/corpus.jsonl")
    require(len(corpus) == manifest["passages"], "Corpus count mismatch")
    results = []
    for stage in stages:
        results.append(audit_extractions(run, corpus, require_complete=require_complete)
                       if stage == "extraction" else audit_paths(
                           run, require_complete=require_complete))
    return {
        "status": "passed", "scope": "response_artifact_provenance_only",
        "run": str(run), "dataset": config["dataset"], "observed_unix": time.time(),
        "require_complete": require_complete, "new_model_calls": 0, "gold_read": False,
        "helper_sha256": sha256(Path(__file__)), "stages": results,
        "limitations": [
            "Checks exact request reconstruction, saved response identity, zero reported "
            "reasoning tokens and artifact equality, not factual extraction/question quality.",
            "Partial saved-file audits do not establish full coverage or complete reproduction.",
            "Final dataset acceptance additionally requires aliases/graph/index verification, "
            "all predictions and ID-aligned evaluation through verify_run.",
            "Cached responses are local evidence; this does not establish remote billed usage.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--stages", nargs="+", choices=("extraction", "path_questions"),
                        default=["extraction"])
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--save", type=Path)
    args = parser.parse_args()
    from reproduction.helpers.storage_paths import checked_run_path
    run = checked_run_path(args.run)
    require(not args.save or args.save.resolve().is_relative_to(run.resolve()),
            "Saved evidence must remain inside the verified current run")
    result = audit(run, args.stages, require_complete=args.require_complete)
    if args.save:
        write_json(args.save, result)
    printable = {**result, "stages": [
        {k: v for k, v in s.items() if k != "artifacts"} for s in result["stages"]]}
    print(json.dumps(printable, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
