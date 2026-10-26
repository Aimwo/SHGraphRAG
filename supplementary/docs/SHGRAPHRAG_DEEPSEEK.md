# SHGraphRAG with DeepSeek

[Home](../../README.md) · [Quick start](QUICKSTART.md) · [Implementation boundaries](LIMITATIONS.md)

This profile preserves the source variants used for the successful DeepSeek runs. It uses upstream graph construction and the supplied tuple-based QA pipeline. Its two-hop path lookup runs online; **an offline path-question repository was not built: 0 entries, 0 bytes**. This differs from the GPT SHGraphRAG implementation and the full offline repository described in the manuscript.

## Source identity

| Dataset | Published source | Final historical execution-bundle identifier |
| --- | --- | --- |
| HotpotQA | [`hotpotqa/`](../code/shgraphrag_deepseek/hotpotqa/) | `ad52b94b6effd9bd1e4ffc9bd810f5028c39f013e17609fb7f6c5c05cad43daf` |
| 2Wiki | [`2wiki/`](../code/shgraphrag_deepseek/2wiki/) | `c220524c775ade9caaed3e95cf1eac1776e6a78911783303ef8143be7656f40e` |
| MuSiQue | [`musique/`](../code/shgraphrag_deepseek/musique/) | `e9f456bc59c793965f2d92f241a404d9a464c105a508fa4f1e524994dc4e9eb6` |

These identifiers hash the original source manifest, not the ZIP bytes. Every selected exported file was checked against the corresponding archive manifest. Publication changes have separate before/after hashes.

## 1. Install the runtime and prepare inputs

Use Python 3.12.13 and the selected dataset's runtime requirements and constraints. Follow [DATA.md](DATA.md) using the common preparation environment, then select this runtime:

```powershell
$python = (Resolve-Path supplementary/.venv-sh-deepseek/Scripts/python.exe).Path
$dataRoot = 'D:/research/shgraphrag-prepared'
& $python supplementary/tools/reproduce.py --profile sh-deepseek --dataset hotpotqa --data-root $dataRoot --run-name reproduce_01 --phase prepare --execute
```

This freezes all 1,000 seed42 IDs and validates the real `DatasetProcessor` against the complete development source. It checks every selected question and every candidate chunk in draw order. The inference processor receives the full dev Parquet through `SHGRAPH_FULL_DEV_PARQUET`. The 1,000-row handoff is not sampled again.

## 2. Prepare an owned graph database

The following explicit service step targets only this dataset's isolated source copy. On Windows:

```powershell
$env:JAVA_HOME = 'C:/Program Files/YourJava17Installation'
Push-Location supplementary/code/shgraphrag_deepseek/hotpotqa
& $python -m reproduction.prepare_services
& $python -m reproduction.database
Pop-Location
```

Use an actual Java 17 installation path. The public downloader obtains Neo4j 5.26.0, GDS 2.13.4, and APOC 5.26.0 into `.runtime/`. Startup refuses ports belonging to another instance. The ownership guard verifies the isolated home, PID, creation time, and exact loopback URI before graph operations.

The graph build invokes database clearing. Keep the owned instance isolated, and do not replace its destination with an existing research or application database. For 2Wiki or MuSiQue, change the source subdirectory; their fixed port pairs are listed in [ENVIRONMENT.md](ENVIRONMENT.md).

Also provide local BGE‑M3 at port 11434. No model or embedding service was started during publication validation.

## 3. Inspect the model contract

Each `reproduction/relay_config.json` records:

```json
{
  "provider": "deepseek-relay",
  "base_url": "https://new.21r.cn/v1",
  "model": "deepseek-v4.1-flash",
  "generation_parameters": {}
}
```

The route is a historical third-party relay. The requested/returned model allowlist is checked. The empty generation object means temperature, max tokens, and reasoning/thinking controls were omitted; the provider's effective defaults are unknown. Provide your own `DEEPSEEK_API_KEY` for that explicitly selected route. If you use another provider/model, record it as a new condition.

## 4. Run the pipeline

```powershell
& $python supplementary/tools/reproduce.py --profile sh-deepseek --dataset hotpotqa --data-root $dataRoot --run-name reproduce_01 --phase run --execute --allow-model-api
```

Run each dataset from its own source variant. The driver performs owned-database checks, snapshots source, validates the frozen sample, executes construction, confirms indexes are online, runs QA, and evaluates every frozen ID.

HotpotQA and MuSiQue use dispatch 32 and physical text gate 64. 2Wiki's final source uses dispatch/gate 100 and embedding concurrency 8, preserving its earlier capacity-transition audit. A single per-question prompt is not combined with another question's prompt. The shared embedding batch limit is up to 32 texts.

Retrieval takes the global top 10 vector candidates and then filters by question ID; entity IDs also include the question namespace. This method execution must not be described as the same global retrieval experiment as the HyperGraphRAG DeepSeek profile.

## 5. Score and verify

```powershell
& $python supplementary/tools/reproduce.py --profile sh-deepseek --dataset hotpotqa --data-root $dataRoot --run-name reproduce_01 --phase evaluate --execute
& $python supplementary/tools/reproduce.py --profile sh-deepseek --dataset hotpotqa --data-root $dataRoot --run-name reproduce_01 --phase verify --execute
```

Evaluation executes the native answer scorer over all frozen IDs. 2Wiki uses official answer-ID aliases/demonyms; MuSiQue uses native answer aliases. See [EVALUATION.md](EVALUATION.md) for single-reference versus native scores.

Saved artifacts include:

| Location | Contents |
| --- | --- |
| `reproduction/artifacts/<dataset>_seed42_n1000/` | Frozen IDs, source indices, selected Parquet, native gold, corpus/mappings, manifests |
| `reproduction/runs/<run-name>/source/` | Source ZIP and hashes captured for that execution |
| `effective_config.json`, `state.json` | Provider, sample, scheduling, source identity, completion |
| Construction / extraction / merge checkpoints | Question/passage alignment and resumable build state |
| `qa/`, `metrics.json`, `final_summary.json` | Predictions, full-denominator metrics, and run aggregates |
| Request events / usage / resource logs | Physical-attempt provenance and observed timing |

Some legacy test files refer to a retired provider and are retained as historical source only. The public `offline_tests.py` runs the applicable DeepSeek, sampling, embedding, merge-checkpoint, and index-readiness suites and excludes the obsolete provider-specific `test_runtime.py`.
