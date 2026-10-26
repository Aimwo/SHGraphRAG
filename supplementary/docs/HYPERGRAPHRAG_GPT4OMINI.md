# HyperGraphRAG with GPT‑4o mini

[Home](../../README.md) · [Quick start](QUICKSTART.md) · [Configuration](CONFIGURATION.md)

This profile uses vendored HyperGraphRAG, OpenRouter GPT‑4o mini, and the same seed20261020 questions and candidate unions as the GPT SHGraphRAG profile.

## Source and accepted run

- Source: [`code/hypergraphrag_gpt4omini/`](../code/hypergraphrag_gpt4omini/).
- Upstream: [LHRLAB/HyperGraphRAG](https://github.com/LHRLAB/HyperGraphRAG), commit `d587cdf8c3fe2be7719557f845324cb3a321f5e2`.
- Frozen successful run: `full1000_gpt4omini_20261022_overflow1`.
- Acceptance: 3,000 prediction checkpoints and 73 acceptance checks, completed October 22, 2026.
- Evidence: [`records/hypergraphrag_gpt4omini/`](../records/hypergraphrag_gpt4omini/).

The separate similarly named `overflowfallback1` attempt is not selected. Source files were exported from the accepted run's own `source_snapshot`, not from an unversioned latest working directory.

## 1. Install and prepare

Use Python 3.11 and [`revision/requirements.lock.txt`](../code/hypergraphrag_gpt4omini/revision/requirements.lock.txt). Prepare the `o200k_base` tokenizer cache, then reconstruct data as in [DATA.md](DATA.md).

```powershell
$python = (Resolve-Path supplementary/.venv-hyper-gpt/Scripts/python.exe).Path
$dataRoot = 'D:/research/shgraphrag-prepared'
& $python supplementary/tools/reproduce.py --profile hyper-gpt4omini --dataset hotpotqa --data-root $dataRoot --run-name reproduce_01 --phase prepare --execute
```

Preparation checks a shared `datasets.lock.json`, the ordered IDs, candidate corpus count and hash, and query fields. No gold or support annotations enter the transport.

## 2. Start the isolated embedding service

On Windows, with Ollama and the pinned BGE‑M3 already installed:

```powershell
& supplementary/code/hypergraphrag_gpt4omini/monitoring/start_isolated_embedding.ps1 -Port 11436
```

The launcher refuses an occupied port, starts a hidden separate Ollama process, writes an ownership/service record, and restores the parent process environment. It uses the existing model store; it does not pull, replace, or delete models.

The embedding request uses `num_ctx=8192`, `num_batch=8192`, and `truncate=false`. An explicit context-overflow error is the only trigger for a separately recorded `truncate=true` fallback. All fallback decisions and payload identities are audited. This is part of the selected completed run's configuration.

## 3. Execute the three datasets

Provide your own authorized OpenRouter credential. Execute all three datasets under one frozen campaign:

```powershell
& $python supplementary/tools/reproduce.py --profile hyper-gpt4omini --dataset all --data-root $dataRoot --run-name reproduce_01 --phase run --execute --allow-model-api
```

The driver preserves completed datasets when resuming. The campaign uses 32 workers by default and a frozen source/config/input contract, including the dataset selection. Do not switch a single-dataset campaign to another dataset under the same name. The wrapper's `--dataset all` expands to this native command after setting the data root and pinned provider environment:

```powershell
& $python -m revision_openrouter.run --name reproduce_01 --limit 1000 --workers 32 --datasets hotpotqa 2wiki musique
```

That native command must run from `supplementary/code/hypergraphrag_gpt4omini/`; prefer the public wrapper so directory and environment setup are explicit and reproducible.

## 4. Understand the method execution

HyperGraphRAG tokenizes each pooled candidate passage, uses 1,200-token chunks with overlap 100, extracts hyper-relations/entities with two gleaning rounds, and builds its native stores. Hybrid retrieval uses top-k 60 and 4,000-token section budgets. The answer adapter returns the short-answer schema required by the evaluation.

Prompt wording is retained in [`hypergraphrag/prompt.py`](../code/hypergraphrag_gpt4omini/hypergraphrag/prompt.py). Provider identity checks, exact-response cache keys, finish reasons, transient retries, refusal handling, response usage, and query attempts are recorded in `revision_openrouter/`.

Filtered, truncated, or incomplete extraction output is not silently promoted into a complete graph. A terminal unavailable response can lead to a retained empty answer; the full evaluation denominator remains 1,000.

## 5. Evaluate and verify

```powershell
& $python supplementary/tools/reproduce.py --profile hyper-gpt4omini --dataset hotpotqa --data-root $dataRoot --run-name reproduce_01 --phase evaluate --execute
& $python supplementary/tools/reproduce.py --profile hyper-gpt4omini --dataset hotpotqa --data-root $dataRoot --run-name reproduce_01 --phase verify --execute
```

The campaign verifier expects all three dataset directories under the same name. Complete them before the final `verify` command; `--dataset` identifies the wrapper profile, not a restriction of the historical campaign-level audit.

Artifacts live under `revision_openrouter/runs/<run-name>/`:

- `provenance.json`, `source_snapshot/`, `datasets.lock.json`: source, configuration, and input locks.
- Per-dataset graph/index files, prediction checkpoints, metrics, and summary.
- Request/cache/embedding attempt logs and immutable response artifacts.
- Construction attempts, per-query timing, resource records, and independent verification.

Cold one-pass construction time is not recovered by dividing wall time by the number of workers. The historical campaign included verified same-model cache recovery; preserve that fact when using its aggregate efficiency records.
