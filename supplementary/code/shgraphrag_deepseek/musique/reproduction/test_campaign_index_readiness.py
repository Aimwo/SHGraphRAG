"""Offline regression for resuming after asynchronous vector-index creation."""

import ast
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import tempfile
import time
from types import ModuleType
import unittest
from unittest.mock import Mock, patch


class ReachedQA(Exception):
    pass


class IndexReadinessTests(unittest.TestCase):
    def run_checkpointed_construction(self, root, *, timeout=False, stays_pending=False):
        # Compile only the real orchestration function. Tests cannot initialize
        # a model, load credentials, contact Ollama, or connect to a database.
        source = Path(__file__).with_name("campaign.py").read_text(encoding="utf-8")
        function = next(
            node for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef) and node.name == "run_full_pipeline"
        )
        module = ast.Module(body=[function], type_ignores=[])
        modules = {}
        parent = ModuleType("build")
        parent.__path__ = []
        modules["build"] = parent
        for name, constructor in [
            ("build.build_graph", "KnowledgeGraphBuilder"),
            ("build.build_index_and_community", "IndexCommunityBuilder"),
            ("build.build_chunk_index", "ChunkIndexBuilder"),
        ]:
            item = ModuleType(name)
            setattr(item, constructor, Mock(side_effect=AssertionError("Checkpoint was rebuilt")))
            modules[name] = item
        graph = Mock()
        ready = False

        def query(statement, *args, **kwargs):
            nonlocal ready
            if statement.startswith("CALL db.awaitIndexes"):
                if timeout:
                    raise TimeoutError("Index population did not finish within the bound")
                ready = not stays_pending
                return []
            if statement.startswith("MATCH (n:"):
                return [{
                    "total": 5, "embedded": 5,
                    "min_dimension": 1024, "max_dimension": 1024,
                }]
            raise AssertionError(f"Unexpected database query: {statement}")

        graph.query.side_effect = query

        def snapshot(_graph):
            return {"vector_indexes": [
                {"name": "entity", "state": "ONLINE"},
                {"name": "chunk", "state": "ONLINE" if ready else "POPULATING"},
            ]}

        @contextmanager
        def stage(_name):
            yield

        namespace = {
            "time": time,
            "graph_snapshot": snapshot,
            "validate_graph_alignment": lambda _graph: {"passed": True},
            "utc_now": lambda: "offline-test",
            "write_json": lambda path, data: path.write_text(json.dumps(data), encoding="utf-8"),
        }
        exec(compile(module, str(Path(__file__).with_name("campaign.py")), "exec"), namespace)
        pipeline = Mock()
        pipeline.run_on_dataset.side_effect = ReachedQA
        for name in [
            "base_graph_completed.json", "entity_index_completed.json",
            "chunk_index_completed.json",
        ]:
            (root / name).write_text("{}", encoding="utf-8")
        with patch.dict(sys.modules, modules):
            try:
                namespace["run_full_pipeline"](root, graph, pipeline, stage, {})
            finally:
                self.pipeline = pipeline
                self.graph = graph
                for name, item in modules.items():
                    if name != "build":
                        next(value for value in vars(item).values() if isinstance(value, Mock)).assert_not_called()

    def test_populating_index_is_awaited_before_resuming_qa(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(ReachedQA):
                self.run_checkpointed_construction(root)
            self.assertTrue((root / "construction_summary.json").exists())
            self.pipeline.run_on_dataset.assert_called_once_with()

    def test_wait_timeout_keeps_qa_stopped_and_construction_unpublished(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(TimeoutError):
                self.run_checkpointed_construction(root, timeout=True)
            self.assertFalse((root / "construction_summary.json").exists())
            self.pipeline.run_on_dataset.assert_not_called()

    def test_indexes_still_pending_after_wait_fail_the_existing_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "Vector indexes are not online"):
                self.run_checkpointed_construction(root, stays_pending=True)
            self.assertFalse((root / "construction_summary.json").exists())
            self.pipeline.run_on_dataset.assert_not_called()


if __name__ == "__main__":
    unittest.main()
