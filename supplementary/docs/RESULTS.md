# Recorded completed-run results

[Home](../../README.md) · [Evaluation definitions](EVALUATION.md) · [Limitations](LIMITATIONS.md)

The following values are historical aggregates from completed revision campaigns. Each row covers 1,000 development questions. Values are percentages; rounding is display-only. The evidence files retain the unrounded values.

## GPT‑4o mini: common single-reference protocol

Seed **20261020**, OpenRouter `openai/gpt-4o-mini`, local BGE‑M3.

| Dataset | Method | Primary EM | Primary F1 |
| --- | --- | ---: | ---: |
| HotpotQA | SHGraphRAG | 53.00 | 67.18 |
| HotpotQA | HyperGraphRAG | 53.10 | 69.13 |
| 2Wiki | SHGraphRAG | 58.10 | 65.40 |
| 2Wiki | HyperGraphRAG | 42.60 | 48.52 |
| MuSiQue | SHGraphRAG | 27.40 | 38.88 |
| MuSiQue | HyperGraphRAG | 22.70 | 33.79 |

Evidence: [SHGraphRAG aggregates](../records/shgraphrag_gpt4omini/) and [HyperGraphRAG aggregates](../records/hypergraphrag_gpt4omini/). This primary protocol applies HotpotQA-style single-reference answer scoring across all three datasets.

## GPT‑4o mini: native alias-aware protocol

The predictions are the same as above; only the evaluation reference/normalization policy changes.

| Dataset | Method | Native EM | Native F1 |
| --- | --- | ---: | ---: |
| HotpotQA | SHGraphRAG | 53.00 | 67.18 |
| HotpotQA | HyperGraphRAG | 53.10 | 69.13 |
| 2Wiki | SHGraphRAG | 63.00 | 69.41 |
| 2Wiki | HyperGraphRAG | 45.90 | 51.56 |
| MuSiQue | SHGraphRAG | 30.60 | 40.79 |
| MuSiQue | HyperGraphRAG | 25.30 | 35.17 |

## DeepSeek: native answer protocol

Seed **42**, `deepseek-v4.1-flash` on the historical third-party relay, local BGE‑M3.

| Dataset | Method | Native EM | Native F1 |
| --- | --- | ---: | ---: |
| HotpotQA | SHGraphRAG | 60.80 | 76.22 |
| HotpotQA | HyperGraphRAG | 64.70 | 79.46 |
| 2Wiki | SHGraphRAG | 75.50 | 82.75 |
| 2Wiki | HyperGraphRAG | 72.60 | 77.00 |
| MuSiQue | SHGraphRAG | 49.60 | 62.00 |
| MuSiQue | HyperGraphRAG | 46.90 | 58.41 |

Evidence: [SHGraphRAG metrics](../records/shgraphrag_deepseek/), [HyperGraphRAG summaries](../records/hypergraphrag_deepseek/), and [independent comparison audit](../records/deepseek_comparison/).

The DeepSeek rows share question IDs and candidate passages within each dataset. SHGraphRAG's question-scoped retrieval and HyperGraphRAG's global retrieval remain different experimental conditions. The SH DeepSeek implementation contains no offline path-question repository.

## Canonical fields remain separate

For example, DeepSeek 2Wiki SHGraphRAG canonical EM/F1 are **69.30 / 78.37**, while native alias-aware values are **75.50 / 82.75**. DeepSeek HyperGraphRAG canonical values are **66.80 / 72.97**, while native values are **72.60 / 77.00**. Do not silently replace one protocol with another in a table.

DeepSeek HyperGraphRAG's original 1,000-question passes include normal empty answers: 19 HotpotQA, 161 2Wiki, and 114 MuSiQue. Later empty-answer retry supplements are not merged into the table above. GPT empty-answer counts are also retained in each historical summary.

## Graph and path evidence

| Profile | Offline path-question repository |
| --- | --- |
| SH GPT | 1,064,666 / 360,444 / 2,499,468 questions for Hotpot / 2Wiki / MuSiQue |
| SH DeepSeek | 0 entries; online two-hop lookup |
| Hyper GPT / DeepSeek | Not applicable to the baseline architecture |

Additional graph scale, request counts, token usage, storage, and observed latency are available in the aggregate audit records. A complete independent cold one-pass construction time and independent HyperGraphRAG index-creation time were not recovered. No missing timing is filled with resumed wall time or worker-adjusted arithmetic.

Cross-backbone score changes are not isolated model effects: the samples, SHGraphRAG implementations, retrieval scopes, generation defaults, and cache histories also differ. These tables do not reproduce the submitted Gemini-era rows.
