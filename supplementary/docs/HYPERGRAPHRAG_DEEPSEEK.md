# HyperGraphRAG with DeepSeek

[Home](../../README.md) · [Quick start](QUICKSTART.md) · [Configuration](CONFIGURATION.md)

This profile uses the exact successful per-dataset source snapshots, local BGE‑M3, and seed42 inputs matched to the DeepSeek SHGraphRAG runs. Its retrieval searches the global candidate union. The evaluated implementation includes an explicit summary adaptation for the relay model.

## Selected campaigns

The accepted three-dataset controller completed on October 25, 2026. Published source comes from these final execution directories:

| Dataset | Frozen successful directory |
| --- | --- |
| HotpotQA | `hypergraphrag_deepseek_bge_hotpotqa_seed42_n1000_c100_summary500_budget8192` |
| 2Wiki | `hypergraphrag_deepseek_bge_2wiki_seed42_n1000_c100_summary500_budget8192` |
| MuSiQue | `hypergraphrag_deepseek_bge_musique_seed42_n1000_c100_summary500_budget8192_runtimefix1` |

Sources are under [`code/hypergraphrag_deepseek/`](../code/hypergraphrag_deepseek/). Each retains the upstream algorithm, full experiment harness, original prompts, `provenance/source.json`, upstream `algorithm.patch`, and the dependency lock.

## 1. Install, reconstruct, and prepare

Use Python 3.11 and the selected dataset's `requirements.lock.txt`. Prepare `o200k_base` in the chosen tokenizer cache, reconstruct official inputs, and execute:

```powershell
$python = (Resolve-Path supplementary/.venv-hyper-deepseek/Scripts/python.exe).Path
$dataRoot = 'D:/research/shgraphrag-prepared'
& $python supplementary/tools/reproduce.py --profile hyper-deepseek --dataset hotpotqa --data-root $dataRoot --phase prepare --execute
```

The public preparation adapter copies only independently reconstructed inputs and verifies every hash, exact ordered IDs, 1,000-question denominator, candidate count, and query field boundary.

The final offline readiness check can report `api_calls_paused_by_user` because no execution opt-in is set. Its remaining source/input/config checks must still pass. Read the `blockers` list rather than treating that status as a failed data reconstruction.

## 2. Use the historical provider and embedding contract

`experiment/config.json` records `deepseek-v4.1-flash` on the same historical relay as the SH DeepSeek profile, empty general generation parameters, and local BGE‑M3 at port 11434. Provide your own authorized `DEEPSEEK_API_KEY`.

The physical request ceiling is 100, shared by text and embedding calls. Embeddings have an additional four-request ceiling and 32-text batch size. Do not mistake dispatch workers for observed simultaneous physical calls.

No Neo4j service is used by this profile.

## 3. Execute

```powershell
& $python supplementary/tools/reproduce.py --profile hyper-deepseek --dataset hotpotqa --data-root $dataRoot --phase run --execute --allow-model-api
```

Repeat preparation and execution with `--dataset 2wiki` and `--dataset musique`. Each dataset has its own `inputs/` and `run/` below its frozen implementation. `--run-name` does not create another Hyper DeepSeek run; use a fresh checkout for an independent repeat.

The worker lock and saved provenance reject concurrent workers or mismatched code/configuration. To pause intentionally, create `STOP_MODEL_CALLS` in that dataset's implementation root. The transport checks it before new physical calls; preserve completed artifacts and remove the marker only when intentionally resuming.

## 4. Preserve the summary adaptation

The native retrieval settings remain chunk 1,200 / overlap 100 / gleaning 2 / hybrid top-k 60 / context sections 4,000. The summary path uses:

1. A prompt requesting at most 500 visible tokens.
2. A physical generation budget of 8,192 tokens, allowing the relay's reasoning/output behavior.
3. Acceptance of nonempty length-limited summary output where the source policy permits it.
4. A valid visible-text prefix of at most 500 `o200k_base` tokens when clipping is needed.
5. Preservation of the available description for a terminal unavailable summary.
6. An audit of original/adopted output identities, finish reason, token counters, and policy decisions.

[`experiment/summary_policy.py`](../code/hypergraphrag_deepseek/hotpotqa/experiment/summary_policy.py) and `experiment/transport.py` contain the exact rules. Do not describe the adapted driver as untouched upstream HyperGraphRAG.

The public fresh-run contract imports **no private construction, response, capacity-recovery, or summary-validation cache**. Historical import records are evidence about the completed source campaign, not a dependency of a new public run.

## 5. Evaluate and verify

```powershell
& $python supplementary/tools/reproduce.py --profile hyper-deepseek --dataset hotpotqa --data-root $dataRoot --phase evaluate --execute
& $python supplementary/tools/reproduce.py --profile hyper-deepseek --dataset hotpotqa --data-root $dataRoot --phase verify --execute
```

The pipeline also invokes independent verification after construction and QA. It checks source snapshots, input IDs, graph/index hashes and counts, immutable query attempts, provider identities, physical request artifacts, summary policy, all 1,000 prediction IDs, and native/canonical metrics.

Normal empty answers remain in the denominator. The historical first complete 1,000-question pass is the table source; a later empty-answer retry supplement is not merged into that table.

Important outputs under each dataset's `run/` include `provenance.json`, `source_snapshot/`, graph/index stores, `index_complete.json`, `predictions/`, query and physical request artifacts, `evaluation/metrics.json`, `summary.json`, and `verification.json`. Preserve all of them when reporting a new execution; this repository includes aggregate historical evidence only.
