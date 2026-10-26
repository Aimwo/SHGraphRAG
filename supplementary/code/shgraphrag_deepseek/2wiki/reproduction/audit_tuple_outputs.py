"""Audit saved extraction against the unchanged upstream tuple regexes."""

import json
import pickle
import re

from reproduction.campaign import DEFAULT_RUN
from reproduction.runtime import utc_now, write_json
from model.deepseek_relay import active_refusal_chunk_hashes

NODE = re.compile(r'\("entity" : "(.+?)" : "(.+?)" : "(.+?)"\)')
RELATION = re.compile(r'\("relationship" : "(.+?)" : "(.+?)" : "(.+?)" : "(.+?)" : (.+?)\)')


def main():
    refused = active_refusal_chunk_hashes(DEFAULT_RUN)
    rows = []
    for path in (DEFAULT_RUN / "cache/graph").glob("*.pkl"):
        try:
            with path.open("rb") as stream:
                text = pickle.load(stream)
        except EOFError:
            continue
        nodes = NODE.findall(text)
        relationships = RELATION.findall(text)
        bad_weights = 0
        for relation in relationships:
            try:
                float(relation[-1])
            except ValueError:
                bad_weights += 1
        rows.append({
            "chunk_hash": path.stem,
            "provider_refused": path.stem in refused,
            "parsed_entity_mentions": len(nodes),
            "parsed_relationship_mentions": len(relationships),
            "unparseable_relation_weights": bad_weights,
            "entity_marker_present": '"entity"' in text or '"实体"' in text,
        })
    report = {
        "generated_at": utc_now(),
        "cached_unique_passages": len(rows),
        "provider_refused_unique_passages": sum(row["provider_refused"] for row in rows),
        "zero_entity_chunk_hashes": [
            row["chunk_hash"] for row in rows if not row["parsed_entity_mentions"]
        ],
        "zero_entities_with_entity_marker": sum(
            not row["parsed_entity_mentions"] and row["entity_marker_present"] for row in rows
        ),
        "entity_mentions": sum(row["parsed_entity_mentions"] for row in rows),
        "relationship_mentions": sum(row["parsed_relationship_mentions"] for row in rows),
        "unparseable_relation_weights": sum(row["unparseable_relation_weights"] for row in rows),
        "per_chunk": rows,
        "note": (
            "Original parser matching only. Some zero-entity responses contain malformed "
            "entity tuples. Original parsing behavior is preserved; no format repair or "
            "additional model requests were performed by this audit."
        ),
    }
    write_json(DEFAULT_RUN / "tuple_parse_audit.json", report)
    print(json.dumps({
        key: value for key, value in report.items()
        if key not in ("per_chunk", "zero_entity_chunk_hashes")
    }, indent=2))


if __name__ == "__main__":
    main()
