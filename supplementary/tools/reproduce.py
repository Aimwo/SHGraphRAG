"""Plan or execute one public reproduction profile.

Planning uses only the standard library. No credentials are read by this tool.
Model execution requires both --execute and --allow-model-api.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

SUPPLEMENT = Path(__file__).resolve().parents[1]
PROFILES = ("sh-gpt4omini", "hyper-gpt4omini", "sh-deepseek", "hyper-deepseek")
DATASETS = ("hotpotqa", "2wiki", "musique")


def implementation(profile, dataset):
    mapping = {
        "sh-gpt4omini": "shgraphrag_gpt4omini",
        "hyper-gpt4omini": "hypergraphrag_gpt4omini",
        "sh-deepseek": f"shgraphrag_deepseek/{dataset}",
        "hyper-deepseek": f"hypergraphrag_deepseek/{dataset}",
    }
    return SUPPLEMENT / "code" / mapping[profile]


def command_plan(profile, dataset, data_root, run_name, phase, python, workers=None):
    if dataset == "all":
        if profile != "hyper-gpt4omini":
            raise ValueError("The all-dataset campaign selector is supported by Hyper GPT only")
        plans = [command_plan(profile, item, data_root, run_name, phase, python, workers)
                 for item in DATASETS]
        plan = plans[0]
        plan["dataset"] = "all"
        if phase == "run":
            command = plan["commands"][0]
            command[command.index("--datasets") + 1:] = list(DATASETS)
        elif phase in ("prepare", "evaluate"):
            plan["commands"] = [command for item in plans for command in item["commands"]]
        return plan
    root = implementation(profile, dataset)
    environment = {
        "SHGRAPHRAG_DATA_ROOT": str(data_root), "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONHASHSEED": "42" if "deepseek" in profile else "0",
        "NO_PROXY": "localhost,127.0.0.1,::1",
    }
    if "gpt4omini" in profile:
        environment.update(OPENROUTER_BASE_URL="https://openrouter.ai/api/v1",
                           OPENROUTER_MODEL="openai/gpt-4o-mini")
    commands = []

    def module(name, *args):
        commands.append([python, "-X", "utf8", "-u", "-m", name, *map(str, args)])

    def script(code, *args):
        commands.append([python, "-X", "utf8", "-c", code, *map(str, args)])

    if profile == "sh-gpt4omini":
        run = root / "reproduction/runs" / run_name / dataset
        count = 64 if workers is None else workers
        if not 1 <= count <= 64:
            raise ValueError("SH GPT profile requires 1..64 workers")
        if phase in ("prepare", "run"):
            module("reproduction.runner", "--dataset", dataset, "--source", data_root,
                   "--output", run, "--limit", 1000, "--workers", count,
                   "--stage", "prepare" if phase == "prepare" else "all")
        elif phase == "evaluate":
            script("import json,sys; from pathlib import Path; "
                   "from reproduction.evaluate import evaluate_run; "
                   "print(json.dumps(evaluate_run(Path(sys.argv[1])), indent=2))", run)
        elif phase == "verify":
            module("reproduction.verify_run", run)
    elif profile == "hyper-gpt4omini":
        run = root / "revision_openrouter/runs" / run_name
        count = 32 if workers is None else workers
        if not 1 <= count <= 32:
            raise ValueError("Hyper GPT profile requires 1..32 workers")
        if phase == "prepare":
            script("import json,sys; from pathlib import Path; "
                   "from revision_openrouter.run import LOCK,read_json,prepare; "
                   "entry=next(e for e in read_json(LOCK)['datasets'] if e['dataset']==sys.argv[1]); "
                   "print(json.dumps(prepare(entry,Path(sys.argv[2]),1000)[2], indent=2))",
                   dataset, run / dataset)
        elif phase == "run":
            module("revision_openrouter.run", "--name", run_name, "--limit", 1000,
                   "--workers", count, "--datasets", dataset)
        elif phase == "evaluate":
            script("import json,sys; from pathlib import Path; "
                   "from revision_openrouter.run import LOCK,PROJECT,read_json; "
                   "from revision_openrouter.evaluate import evaluate_run; "
                   "entry=next(e for e in read_json(LOCK)['datasets'] if e['dataset']==sys.argv[1]); "
                   "print(json.dumps(evaluate_run(Path(sys.argv[2]),PROJECT,entry), indent=2))",
                   dataset, run / dataset)
        elif phase == "verify":
            # The historical campaign verifier requires all three dataset folders.
            module("revision_openrouter.verify", run)
    elif profile == "sh-deepseek":
        if workers is not None:
            raise ValueError("DeepSeek scheduling is pinned per dataset in config/settings.py")
        run = root / "reproduction/runs" / run_name
        environment["SHGRAPH_FULL_DEV_PARQUET"] = str(data_root / "normalized" / f"{dataset}.parquet")
        artifact = root / "reproduction/artifacts" / f"{dataset}_seed42_n1000"
        if dataset != "musique":
            environment["SHGRAPH_BENCHMARK_CONFIG"] = str(
                data_root / "deepseek/shgraphrag" / dataset / "benchmark.json")
        if phase == "prepare":
            freezer = "reproduction.freeze_musique_sample" if dataset == "musique" else "reproduction.freeze_benchmark"
            module(freezer, *(["--validate-only"] if (artifact / "manifest.json").exists() else []))
            module("reproduction.validate_processor_sample")
        elif phase == "run":
            module("reproduction.campaign", "--run-dir", run)
            module("reproduction.report", "--run-dir", run, "--compact")
        elif phase == "evaluate":
            scorer = ("qa/musique_evaluate_v1.0.py" if dataset == "musique"
                      else "qa/benchmark_evaluate.py")
            script("import importlib.util,json,sys; from pathlib import Path; "
                   "from reproduction.runtime import write_json; "
                   "spec=importlib.util.spec_from_file_location('offline_metric',sys.argv[1]); "
                   "m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
                   "run=Path(sys.argv[2]); frozen=Path(sys.argv[3]); "
                   "ids=json.loads((frozen/'ids.json').read_text(encoding='utf-8')); "
                   "metrics=m.evaluate(str(run/'qa/predictions.json'),str(frozen/'gold.jsonl'),ids); "
                   "write_json(run/'metrics.json',metrics); "
                   "print(json.dumps({k:v for k,v in metrics.items() if k!='per_question'},indent=2))",
                   root / scorer, run, artifact)
        elif phase == "verify":
            module("reproduction.verify_completed_run", "--run-dir", run)
    else:
        if workers is not None:
            raise ValueError("Hyper DeepSeek physical request limit is pinned in experiment/config.json")
        run = root / "run"
        if phase == "prepare":
            module("experiment.prepare")
            module("experiment.run", "--check")
        elif phase == "run":
            module("experiment.run")
        elif phase == "evaluate":
            module("experiment.evaluate")
        elif phase == "verify":
            module("experiment.verify")
    return {"profile": profile, "dataset": dataset, "phase": phase,
            "working_directory": str(root), "run_directory": str(run),
            "environment_overrides": environment, "commands": commands,
            "model_calls_allowed": False,
            "note": ("Hyper GPT verify checks all three datasets in the same run name."
                     if profile == "hyper-gpt4omini" else
                     "DeepSeek Hyper uses one isolated run directory per dataset."
                     if profile == "hyper-deepseek" else
                     "Prepare/evaluate/verify are offline; run performs model work.")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--dataset", choices=(*DATASETS, "all"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-name", default="public_reproduction")
    parser.add_argument("--phase", choices=("plan", "prepare", "run", "evaluate", "verify"), default="plan")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-model-api", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_name):
        parser.error("Run name must contain only ASCII letters, digits, underscores or hyphens")
    phase = "run" if args.phase == "plan" else args.phase
    plan = command_plan(args.profile, args.dataset, args.data_root.resolve(),
                        args.run_name, phase, sys.executable, args.workers)
    plan["execution_requested"] = args.execute and args.phase != "plan"
    print(json.dumps(plan, indent=2), flush=True)
    if not plan["execution_requested"]:
        return
    if phase == "run" and not args.allow_model_api:
        parser.error("Model execution requires explicit --allow-model-api; no child was started")
    if not (args.data_root / "preparation_validation.json").is_file():
        parser.error("Run tools/prepare_data.py successfully before executing a profile")
    environment = {**os.environ, **plan["environment_overrides"]}
    # Never inherit an accidental opt-in into offline phases.
    environment.pop("SHGRAPHRAG_ALLOW_MODEL_API", None)
    if phase == "run":
        environment["SHGRAPHRAG_ALLOW_MODEL_API"] = "1"
    root = Path(plan["working_directory"])
    if phase == "prepare" and args.profile == "sh-deepseek":
        (root / "reproduction/audit").mkdir(parents=True, exist_ok=True)
    for command in plan["commands"]:
        subprocess.run(command, cwd=root, env=environment, check=True)


if __name__ == "__main__":
    main()
