<div align="center">

<h1>SHGraphRAG</h1>

<p><strong>Supplementary Experiments &amp; Reproduction</strong></p>

<p>SHGraphRAG · HyperGraphRAG &nbsp; | &nbsp; GPT‑4o mini · DeepSeek</p>

<p>
  <a href="LICENSE"><img src="supplementary/assets/badges/license.svg" alt="License: MIT" height="20"></a>
  <a href="supplementary/docs/ENVIRONMENT.md"><img src="supplementary/assets/badges/python.svg" alt="Recorded Python versions: 3.11 and 3.12" height="20"></a>
  <a href="#experiment-matrix"><img src="supplementary/assets/badges/profiles.svg" alt="Experiment profiles: 4" height="20"></a>
  <a href="supplementary/docs/DATA.md"><img src="supplementary/assets/badges/benchmarks.svg" alt="Benchmarks: 3" height="20"></a>
  <a href="supplementary/docs/ENVIRONMENT.md#bgem3"><img src="supplementary/assets/badges/embedding.svg" alt="Embedding: BGE-M3" height="20"></a>
</p>

<p>
  <a href="supplementary/docs/QUICKSTART.md"><strong>Quick start</strong></a> &nbsp;·&nbsp;
  <a href="supplementary/docs/CONFIGURATION.md">Configuration</a> &nbsp;·&nbsp;
  <a href="supplementary/docs/RESULTS.md">Results</a> &nbsp;·&nbsp;
  <a href="supplementary/docs/PROVENANCE.md">Provenance</a>
</p>

<p><strong>English</strong> &nbsp;|&nbsp; <a href="README.zh-CN.md">简体中文</a></p>

</div>

---

![Reproduction overview: official development releases, two frozen protocols, four implementations, and independent evaluation](supplementary/assets/reproduction-overview.svg)

This branch accompanies *Sparse Hypergraph RAG: Bridging Structural Knowledge for Multi-Hop QA*. It contains the implementations used in the completed additional evaluations on **HotpotQA, 2WikiMultiHopQA, and MuSiQue-Ans**, with **1,000 development questions per dataset and per profile**. The original root implementation is retained; use the versioned implementations under [`supplementary/code/`](supplementary/code/) for these experiments.

> **Scope.** These are revision experiments with alternative model backbones. They do not reproduce the original paper's Gemini scores. GPT‑4o mini and DeepSeek use different frozen samples and, for SHGraphRAG, different implementations. Compare methods within the same documented protocol. [Read the comparability boundaries](supplementary/docs/LIMITATIONS.md) before interpreting the tables.

## Experiment matrix

| Profile | Text model and route used in the recorded run | Seed | Implementation | Reproduction guide |
| --- | --- | ---: | --- | --- |
| `sh-gpt4omini` | `openai/gpt-4o-mini`, OpenRouter Chat Completions | 20261020 | Global sparse graph; offline two-hop path-question repository; exact cosine indexes | [SHGraphRAG / GPT‑4o mini](supplementary/docs/SHGRAPHRAG_GPT4OMINI.md) |
| `hyper-gpt4omini` | `openai/gpt-4o-mini`, OpenRouter Chat Completions | 20261020 | Vendored HyperGraphRAG; hybrid retrieval; audited embedding overflow fallback | [HyperGraphRAG / GPT‑4o mini](supplementary/docs/HYPERGRAPHRAG_GPT4OMINI.md) |
| `sh-deepseek` | `deepseek-v4.1-flash`, historical third-party relay | 42 | Upstream graph construction plus supplied QA; question-scoped retrieval; online two-hop lookup; **0 offline path questions** | [SHGraphRAG / DeepSeek](supplementary/docs/SHGRAPHRAG_DEEPSEEK.md) |
| `hyper-deepseek` | `deepseek-v4.1-flash`, same historical relay | 42 | Vendored HyperGraphRAG; hybrid retrieval; visible-summary limit 500 / generation budget 8,192 | [HyperGraphRAG / DeepSeek](supplementary/docs/HYPERGRAPHRAG_DEEPSEEK.md) |

All profiles use local **BGE‑M3, 1,024 dimensions**. Exact model identity, storage, generation settings, retry policies, prompts, and concurrency are documented separately for each profile. The DeepSeek route was `https://new.21r.cn/v1`; this is recorded provenance, not a claim about an official DeepSeek service or its future availability.

## What is included

- Complete selected implementations: graph construction, entity merging, retrieval, QA, local embedding adapters, evaluation, checkpointing, and run verification.
- Frozen per-dataset DeepSeek source variants, dependency locks, original prompts, explicit public configuration, and publication portability diffs.
- Six ordered question-ID lists and release hashes, allowing independent reconstruction from official datasets.
- Data-free historical configuration, aggregate metrics, and completion receipts, clearly distinguished from publication-time checks.
- A shared data reconstruction script, an inspectable execution planner, offline regression checks, and locally stored SVG diagrams.

**Datasets, answers, predictions, graph databases, indexes, vectors, raw responses, model weights, caches, virtual environments, and credentials are excluded.** Question IDs are small protocol metadata; no question or passage text is bundled.

## Quick start

```bash
git clone --single-branch --branch Sumpplementary https://github.com/Aimwo/SHGraphRAG.git
cd SHGraphRAG
python supplementary/tools/validate_release.py
python supplementary/tools/reproduce.py --profile sh-gpt4omini --dataset hotpotqa --data-root /absolute/path/to/prepared-data
```

The final command prints a plan and starts no model work. For an executable experiment, first install the profile's pinned environment, obtain the official development files, reconstruct the exact sample and candidate corpus, and provide your own explicitly authorized credentials. See the [complete quick start](supplementary/docs/QUICKSTART.md).

Actual model execution requires both `--execute` and `--allow-model-api`. Offline preparation and evaluation do not require a provider key. No source file or launcher loads a private workstation credential store for the supported reproduction entry points.

## Documentation

| Goal | Read |
| --- | --- |
| Set up an environment and execute a profile | [Quick start](supplementary/docs/QUICKSTART.md), [environment reference](supplementary/docs/ENVIRONMENT.md) |
| Obtain and reconstruct the exact evaluation inputs | [Data and sampling protocol](supplementary/docs/DATA.md) |
| Inspect every scientific setting and prompt location | [Configuration reference](supplementary/docs/CONFIGURATION.md) |
| Reproduce graph construction, retrieval, and QA | The four method guides in the experiment matrix |
| Understand EM/F1, aliases, missing answers, and acceptance | [Evaluation and verification](supplementary/docs/EVALUATION.md) |
| Read completed run aggregates and their evidence links | [Results](supplementary/docs/RESULTS.md) |
| Audit exported source hashes and portability changes | [Provenance](supplementary/docs/PROVENANCE.md), [source manifest](supplementary/provenance/source_manifest.json), [portability diff](supplementary/provenance/portability.patch) |
| Understand deviations, reproducibility limits, and failures | [Limitations](supplementary/docs/LIMITATIONS.md), [troubleshooting](supplementary/docs/TROUBLESHOOTING.md) |

## Repository layout

```text
supplementary/
├── assets/                         # Local SVG badges, diagrams, and generators
├── code/
│   ├── shgraphrag_gpt4omini/        # One implementation, three datasets
│   ├── hypergraphrag_gpt4omini/     # One implementation, three datasets
│   ├── shgraphrag_deepseek/         # Dataset-specific frozen source variants
│   └── hypergraphrag_deepseek/      # Dataset-specific frozen source variants
├── docs/                           # Setup, method procedures, reference, audits
├── protocols/                      # Ordered IDs and original input manifests
├── provenance/                     # Source hashes, changes, release validation
├── records/                        # Historical aggregates; no per-example data
└── tools/                          # Reconstruct → plan → run → score → verify
```

The [preserved original README](supplementary/docs/UPSTREAM_README.md) describes the original root implementation. Historical local paths inside provenance identify where evidence originated; they are not required checkout locations. The supported entry point is `supplementary/tools/reproduce.py`.

## Attribution and license

SHGraphRAG code retains the repository's [MIT license](LICENSE). Vendored HyperGraphRAG code retains its [upstream MIT license](supplementary/code/hypergraphrag_gpt4omini/LICENSE) and attribution to Haoran Luo. HyperGraphRAG is pinned to upstream commit `d587cdf8c3fe2be7719557f845324cb3a321f5e2`; see [third-party notices](supplementary/THIRD_PARTY_NOTICES.md).

This supplementary branch is maintained separately from `main`. The source and publication audit spans **October 26–27, 2026**. Run-specific scientific records and the current publication validation have separate provenance.
