"""Count explicit runtime diagnostics from local logs and QA checkpoints."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re

from reproduction.campaign import DEFAULT_RUN
from reproduction.runtime import utc_now, write_json

PATTERNS = {
    "neo4j_query_failed": re.compile(r"^ERROR:HotpotKGQA:Neo4j query failed:"),
    "no_graph_interface": re.compile(r"^ERROR:HotpotKGQA:No graph interface available"),
    "chunk_retrieval_failed": re.compile(r"^Chunk retrieval failed:"),
    "decomposition_parse_fallback": re.compile(
        r"^WARNING:HotpotKGQA:Decompose parsing failed, fallback to single-step"
    ),
    "question_attempt_failed": re.compile(r"^ERROR:HotpotKGQA:QID=.* attempt=\d+ failed:"),
    "tuple_parse_error": re.compile(r"^\[解析错误\]"),
    "graph_document_conversion_error": re.compile(r"^转换GraphDocument时出错"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    root = args.run_dir
    matches = {name: [] for name in PATTERNS}
    logs = {}
    for name in ("stdout.log", "stderr.log", "controller.stdout.log", "controller.stderr.log"):
        path = root / name
        if not path.exists():
            continue
        # Tolerate a live writer's incomplete UTF-8 character at the end.
        content = path.read_text(encoding="utf-8", errors="replace")
        lines = content.splitlines()
        logs[name] = {"bytes": path.stat().st_size, "lines": len(lines)}
        for number, line in enumerate(lines, 1):
            for category, pattern in PATTERNS.items():
                if pattern.search(line):
                    matches[category].append({"log": name, "line": number})
    path = root / "qa/predictions_with_supporting_facts.json"
    predictions = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    answer_states = Counter()
    for row in predictions:
        answer = row.get("answer")
        if not isinstance(answer, str):
            answer_states["non_string"] += 1
        elif not answer.strip():
            answer_states["empty_string"] += 1
        elif answer.strip().lower() in ("none", "unanswerable", "unknown", "null"):
            answer_states["sentinel_string"] += 1
        else:
            answer_states["other_string"] += 1
    report = {
        "generated_at": utc_now(),
        "new_model_api_requests": 0,
        "log_inventory": logs,
        "explicit_diagnostic_line_counts": {
            name: len(rows) for name, rows in matches.items()
        },
        "diagnostic_locations": matches,
        "qa_checkpoint_count": len(predictions),
        "qa_answer_type_counts": dict(answer_states),
        "note": (
            "Counts are occurrences of specific logged diagnostics, not a semantic "
            "correctness test or proof that every possible failure was logged. "
            "Zero query errors does not imply nonempty or accurate retrieval."
        ),
    }
    write_json(root / "execution_log_audit.json", report)
    print(json.dumps({
        "explicit_diagnostic_line_counts": report["explicit_diagnostic_line_counts"],
        "qa_checkpoint_count": len(predictions),
        "qa_answer_type_counts": dict(answer_states),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
