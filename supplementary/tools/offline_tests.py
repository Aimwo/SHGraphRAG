"""Run selected regression suites while prohibiting real socket connections.

Dummy test credentials are not provider keys. All model/embedding fixtures are
local mocks. A tokenizer cache may be supplied as a non-secret dependency cache.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import inspect
import json
import os
from pathlib import Path
import socket
import sys
import unittest
from unittest.mock import patch

from reproduce import implementation, PROFILES, DATASETS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--dataset", choices=DATASETS, default="hotpotqa")
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args()
    data_root = args.data_root.resolve() if args.data_root else None
    root = implementation(args.profile, args.dataset)
    if args.profile == "sh-deepseek":
        for folder in ("reproduction/runs", "reproduction/audit", ".runtime"):
            (root / folder).mkdir(parents=True, exist_ok=True)
    os.chdir(root)
    sys.path.insert(0, str(root))
    os.environ.update(
        SHGRAPHRAG_ALLOW_MODEL_API="1", OPENROUTER_API_KEY="offline-dummy-key",
        DEEPSEEK_API_KEY="offline-dummy-key", OPENROUTER_MODEL="openai/gpt-4o-mini",
        OPENROUTER_BASE_URL="https://openrouter.ai/api/v1",
        OPENROUTER_PROXY="http://127.0.0.1:6789",
        HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
    )
    if data_root:
        os.environ["SHGRAPHRAG_DATA_ROOT"] = str(data_root)
        os.environ["SHGRAPH_FULL_DEV_PARQUET"] = str(
            data_root / "normalized" / f"{args.dataset}.parquet")
        if args.dataset != "musique":
            os.environ["SHGRAPH_BENCHMARK_CONFIG"] = str(
                data_root / "deepseek/shgraphrag" / args.dataset / "benchmark.json")
    else:
        os.environ["SHGRAPHRAG_DATA_ROOT"] = str(root / "data")
    suites = {
        "sh-gpt4omini": ["reproduction.test_reproduction",
                         "reproduction.helpers.test_local_index_transport",
                         "reproduction.helpers.test_index_provenance"],
        "hyper-gpt4omini": ["revision_openrouter.test_revision",
                            "revision_openrouter.test_completion_errors",
                            "revision_openrouter.test_provider_outcomes",
                            "revision_openrouter.test_embedding_client",
                            "revision_openrouter.test_proxy_client",
                            "revision_openrouter.test_evaluation",
                            "revision_openrouter.test_telemetry"],
        "sh-deepseek": ["reproduction.test_sample_selection",
                        "reproduction.test_relay_runtime", "reproduction.test_measured_embeddings",
                        "reproduction.test_merge_checkpoint",
                        "reproduction.test_campaign_index_readiness"],
        "hyper-deepseek": ["experiment.test_experiment",
                           "experiment.test_additional_datasets",
                           "experiment.test_runtime_recovery"],
    }
    modules = suites[args.profile]
    modules = [name for name in modules if (root / (name.replace(".", "/") + ".py")).is_file()]
    # CPython on Windows implements asyncio's internal socketpair with a
    # temporary loopback connection. Permit only that standard-library frame,
    # while blocking external calls and all connections to existing services.
    original_connect = socket.socket.connect

    def guarded_connect(instance, address):
        frames = inspect.stack()
        if any(frame.function == "_fallback_socketpair"
               and Path(frame.filename).resolve() == Path(socket.__file__).resolve()
               for frame in frames):
            return original_connect(instance, address)
        raise AssertionError("Offline validation forbids external/service connections")

    with ExitStack() as stack:
        stack.enter_context(patch("socket.socket.connect", new=guarded_connect))
        for target in ("socket.socket.connect_ex", "socket.create_connection"):
            stack.enter_context(patch(target, side_effect=AssertionError(
                "Offline validation forbids real socket connections")))
        suite = unittest.defaultTestLoader.loadTestsFromNames(modules)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps({"profile": args.profile, "dataset": args.dataset,
                      "tests_run": result.testsRun, "failures": len(result.failures),
                      "errors": len(result.errors), "skipped": len(result.skipped),
                      "external_and_service_connections_permitted": False,
                      "asyncio_internal_socketpair_permitted": True,
                      "status": "passed" if result.wasSuccessful() else "failed"}))
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
