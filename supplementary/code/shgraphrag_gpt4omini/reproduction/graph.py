"""Full hyperedge context, bridge-centered two-hop sparsity, and all simple paths."""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path
import re
import unicodedata

from .client import signature, write_json


def entity_key(name: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", name).casefold().split())


def parse_extraction(text: str) -> dict:
    if "<|COMPLETE|>" not in text:
        raise ValueError("Missing extraction completion marker")
    nodes = {}
    relations = []
    for record in text.split("<|COMPLETE|>")[0].split("##"):
        record = record.strip()
        if not record:
            continue
        if not (record.startswith("(") and record.endswith(")")):
            raise ValueError("Invalid tuple record")
        fields = [p.strip().strip('"').strip() for p in record[1:-1].split("<|>")]
        if fields[0] == "entity" and len(fields) == 4:
            if not fields[1] or not fields[2] or not fields[3]:
                raise ValueError("Empty entity tuple field")
            key = entity_key(fields[1])
            nodes[key] = {"name": fields[1], "type": fields[2],
                          "description": fields[3]}
        elif fields[0] == "relationship" and len(fields) == 6:
            strength = float(fields[5])
            if not 0 <= strength <= 10 or not all(fields[1:5]):
                raise ValueError("Invalid relationship tuple field")
            relations.append({"head": entity_key(fields[1]),
                              "type": fields[2],
                              "tail": entity_key(fields[3]),
                              "description": fields[4], "strength": strength})
        else:
            raise ValueError("Unexpected extraction tuple schema")
    if not nodes:
        raise ValueError("No entities extracted")
    for rel in relations:
        if rel["head"] not in nodes or rel["tail"] not in nodes:
            raise ValueError("Relationship endpoint absent from entity tuples")
    return {"entities": nodes, "relations": relations}


def parse_json_extraction(value: dict) -> dict:
    nodes = {}
    for entity in value["entities"]:
        if not all(isinstance(entity.get(k), str) and entity[k].strip()
                   for k in ("name", "type", "description")):
            raise ValueError("Invalid JSON entity fields")
        nodes[entity_key(entity["name"])] = entity
    if not nodes:
        raise ValueError("No entities extracted")
    relations = []
    missing = sorted({
        rel[field] for rel in value["relationships"] for field in ("source", "target")
        if isinstance(rel.get(field), str) and entity_key(rel[field]) not in nodes})
    if missing:
        raise ValueError("Relationship endpoint absent from entity records: "
                         + json.dumps(missing, ensure_ascii=True))
    for rel in value["relationships"]:
        if not all(isinstance(rel.get(k), str) and rel[k].strip()
                   for k in ("source", "type", "target", "description")):
            raise ValueError("Invalid JSON relationship fields")
        head, tail = entity_key(rel["source"]), entity_key(rel["target"])
        if head not in nodes or tail not in nodes:
            raise ValueError("Relationship endpoint absent from entity records")
        if (not isinstance(rel["strength"], (float, int))
                or isinstance(rel["strength"], bool) or not 0 <= rel["strength"] <= 10):
            raise ValueError("Invalid JSON relationship strength")
        relations.append({"head": head, "tail": tail, "type": rel["type"],
                          "description": rel["description"], "strength": rel["strength"]})
    return {"entities": nodes, "relations": relations}


def assemble(corpus: list[dict], extracted: dict[str, dict]) -> dict:
    entities = {}
    triples = {}
    hyperedges = {}
    for passage in corpus:
        pid = passage["passage_id"]
        data = extracted[pid]
        ids = []
        for key, item in data["entities"].items():
            eid = "entity_" + signature(key)
            ids.append(eid)
            if eid not in entities:
                entities[eid] = {"id": eid, "key": key, "name": item["name"],
                                 "type": item["type"], "descriptions": [], "aliases": [],
                                 "passage_ids": []}
            entity = entities[eid]
            for description in item.get("descriptions", [item["description"]]):
                if description not in entity["descriptions"]:
                    entity["descriptions"].append(description)
            for alias in item.get("aliases", [item["name"]]):
                if alias not in entity["aliases"]:
                    entity["aliases"].append(alias)
            entity["passage_ids"].append(pid)
        hyperedges[pid] = {
            "id": pid, "title": passage["title"], "text": passage["text"],
            "sentences": passage["sentences"], "entity_ids": sorted(set(ids))}
        for rel in data["relations"]:
            head = "entity_" + signature(rel["head"])
            tail = "entity_" + signature(rel["tail"])
            tid = "triple_" + signature([head, rel["type"], tail])
            if tid not in triples:
                triples[tid] = {"id": tid, "head": head, "tail": tail,
                                "type": rel["type"], "descriptions": [],
                                "passage_ids": [], "strengths": []}
            triple = triples[tid]
            if rel["description"] not in triple["descriptions"]:
                triple["descriptions"].append(rel["description"])
            triple["passage_ids"].append(pid)
            triple["strengths"].append(rel["strength"])
    for entity in entities.values():
        entity["passage_ids"] = sorted(set(entity["passage_ids"]))
    for triple in triples.values():
        triple["passage_ids"] = sorted(set(triple["passage_ids"]))
    bridges = {eid for eid, entity in entities.items()
               if len(entity["passage_ids"]) > 1}
    # As in _prune_triples_in_memory_2hop: for each original document,
    # retain edges incident to bridge entities and edges incident to their
    # immediate neighbors. All original hyperedges retain ALL their entities.
    by_passage = defaultdict(list)
    for tid, triple in triples.items():
        for pid in triple["passage_ids"]:
            by_passage[pid].append(tid)
    kept = set()
    for tids in by_passage.values():
        one_hop_nodes = set()
        for tid in tids:
            triple = triples[tid]
            if triple["head"] in bridges or triple["tail"] in bridges:
                one_hop_nodes.update((triple["head"], triple["tail"]))
        for tid in tids:
            triple = triples[tid]
            if triple["head"] in one_hop_nodes or triple["tail"] in one_hop_nodes:
                kept.add(tid)
    retained_entities = {eid for tid in kept for eid in
                         (triples[tid]["head"], triples[tid]["tail"])}
    sparse = {tid: triples[tid] for tid in sorted(kept)}
    stats = {"hyperedges": len(hyperedges), "full_entities": len(entities),
             "full_triples": len(triples), "bridge_entities": len(bridges),
             "sparse_entities": len(retained_entities),
             "sparse_triples": len(sparse),
             "entity_sparsity": 1 - len(retained_entities) / max(1, len(entities)),
             "triple_sparsity": 1 - len(sparse) / max(1, len(triples)),
             "identity": "NFKC_casefold_whitespace_exact_name",
             "sparsification": "supplied_per_document_bridge_two_hop_algorithm"}
    return {"entities": entities, "hyperedges": hyperedges,
            "full_triples": triples, "triples": sparse,
            "bridge_ids": sorted(bridges),
            "retained_entity_ids": sorted(retained_entities), "stats": stats}


def iter_paths(graph: dict):
    """Every simple undirected two-edge path, deduplicated under reversal.

    Relation directions are retained in the two triple records. Parallel
    relation types remain different paths. No degree, count, or length caps.
    """
    adjacency = defaultdict(list)
    triples = graph["triples"]
    for tid, triple in triples.items():
        if triple["head"] == triple["tail"]:
            continue
        adjacency[triple["head"]].append((triple["tail"], tid))
        adjacency[triple["tail"]].append((triple["head"], tid))
    for middle in sorted(adjacency):
        neighbors = sorted(adjacency[middle])
        for (first, tid1), (last, tid2) in combinations(neighbors, 2):
            if first == last:
                continue
            key = [first, tid1, middle, tid2, last]
            sources = sorted(set(triples[tid1]["passage_ids"]
                                 + triples[tid2]["passage_ids"]))
            yield {"id": "path_" + signature(key), "first": first,
                   "middle": middle, "last": last,
                   "triple_ids": [tid1, tid2], "passage_ids": sources}


def path_statistics(graph: dict) -> dict:
    neighbors = defaultdict(lambda: defaultdict(int))
    for triple in graph["triples"].values():
        if triple["head"] == triple["tail"]:
            continue
        neighbors[triple["head"]][triple["tail"]] += 1
        neighbors[triple["tail"]][triple["head"]] += 1
    centers = []
    for middle, counts in neighbors.items():
        degree = sum(counts.values())
        count = degree * (degree - 1) // 2 - sum(
            c * (c - 1) // 2 for c in counts.values())
        centers.append({"entity": graph["entities"][middle]["name"],
                        "degree": degree, "two_edge_paths": count})
    count = sum(c["two_edge_paths"] for c in centers)
    return {
        "two_edge_paths": count,
        "largest_path_centers": sorted(
            centers, key=lambda c: (-c["two_edge_paths"], c["entity"]))[:10],
        "path_vector_bytes_required": count * 2 * 1024 * 4,
        "path_scope": "all_simple_undirected_two_edge_paths",
    }


def path_prompt_item(path: dict, graph: dict) -> dict:
    entities, triples = graph["entities"], graph["triples"]
    return {"id": path["id"],
            "endpoints": [entities[path["first"]]["name"],
                          entities[path["last"]]["name"]],
            "middle": entities[path["middle"]]["name"],
            "facts": [{
                "head": entities[triples[tid]["head"]]["name"],
                "relation": triples[tid]["type"],
                "tail": entities[triples[tid]["tail"]]["name"],
                "description": triples[tid]["descriptions"],
            } for tid in path["triple_ids"]]}
