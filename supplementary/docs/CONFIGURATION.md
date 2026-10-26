# Scientific configuration reference

[Home](../../README.md) · [Environment](ENVIRONMENT.md) · [Prompt sources](#prompt-sources)

The machine-readable historical configurations under [`records/`](../records/) document the completed runs. Executable public configs under [`code/`](../code/) replace private credential files with explicit environment inputs. [`portability.patch`](../provenance/portability.patch) exposes all source changes made for publication.

## Shared protocol boundaries

| Property | GPT‑4o mini pair | DeepSeek pair |
| --- | --- | --- |
| Datasets | HotpotQA distractor / 2Wiki / MuSiQue-Ans | Same dataset releases |
| Split and size | Development, 1,000 each | Development, 1,000 each |
| Sample seed | 20261020 | 42 |
| Text model ID | `openai/gpt-4o-mini` | `deepseek-v4.1-flash` |
| Historical route | `https://openrouter.ai/api/v1` | `https://new.21r.cn/v1`, third-party relay |
| Wire API | Chat Completions | Chat Completions |
| Temperature | Explicit 0 | Omitted from requests |
| Thinking / reasoning parameters | No reasoning parameter | Omitted; provider default is not established |
| Model-level seed | Not established / not sent | Not established / not sent |
| Embeddings | Local BGE‑M3, 1,024 dimensions | Local BGE‑M3, 1,024 dimensions |
| Labels in query files | None | None |
| Full Wikipedia index | No | No |

The SH GPT `EFFORT="none"` compatibility constant appears in artifacts, but its wire request does **not** contain a reasoning parameter. For DeepSeek, an empty `generation_parameters` object must not be described as temperature 0, thinking disabled, or a known deterministic provider setting.

## SHGraphRAG + GPT‑4o mini

Authoritative implementation: [`reproduction/runner.py`](../code/shgraphrag_gpt4omini/reproduction/runner.py), method version `source_alias_merge_global_graph_v2`.

| Setting | Value |
| --- | --- |
| Extraction representation | JSON adapter retaining entities, directed relations, descriptions, strengths |
| Entity-merge candidates | Name cosine > 0.95 or containment |
| LLM alias confirmation | Batches of 10 candidate pairs, followed by global identity reconciliation |
| Sparse construction | Bridge-based sparse graph from reconciled global entities |
| Path enumeration | All simple undirected two-edge paths, uncapped |
| Questions per path | 2 |
| Path generation batch | 32 paths |
| Path retrieval | Top 3 distinct paths |
| Entity retrieval | Top 2 entities, cosine threshold 0.3 |
| Associated passage selection | Top 3, with top 2 fallback/background passages |
| Maximum subquestions | 6 |
| Vector implementation | Normalized float32, exact cosine, stable tie handling |
| Storage | JSON graph + SQLite embedding cache + NumPy arrays |
| Inference scope | Pooled candidate corpus, no question-ID filter |
| Text concurrency | Began at 32; accepted full campaign reached 64 |
| Transport attempts | Up to 5 |
| Structured-output attempts | Up to 3 |
| Endpoint completion attempts | 3, then fixed missing-endpoint records as implemented |
| Embedding batch / truncation | 32 / `truncate=false` |

Generation budgets are stage-specific: extraction, endpoint completion, and path-question batches use up to 8,192 tokens; alias confirmation uses 2,048. The QA stage supplies its own fixed schemas and budgets in `runner.py`. Do not replace these with one global maximum.

## HyperGraphRAG + GPT‑4o mini

Authoritative driver: [`revision_openrouter/run.py`](../code/hypergraphrag_gpt4omini/revision_openrouter/run.py). Upstream algorithm commit: `d587cdf8c3fe2be7719557f845324cb3a321f5e2`.

| Setting | Value |
| --- | --- |
| Chunk size / overlap | 1,200 / 100 tokens |
| Tokenizer | `o200k_base` through the GPT‑4o family |
| Extraction gleaning rounds | 2 |
| Entity summary target | 500 tokens |
| Retrieval mode / top-k | `hybrid` / 60 |
| Context budget | 4,000 tokens per section per branch |
| Storage | NetworkX + NanoVectorDB + JSON KV |
| Inference scope | Pooled candidate corpus |
| Text workers | 32 |
| Output caps | Extract/query-extract/glean 8,192; coverage 64; summary 500; answer 512 |
| Embedding endpoint | `http://127.0.0.1:11436/api/embed` |
| Embedding options | `num_ctx=8192`, `num_batch=8192` |
| Normal embedding request | `truncate=false` |
| Overflow adaptation | Explicit context-overflow error only → separate audited `truncate=true` request |
| Embedding recovery attempts | At most 4 per exact request |
| Text transport retry | Bounded retries for transient transport/provider availability errors |
| Provider terminal outcomes | Filtered, length-limited, missing-finish, and refused responses are recorded separately |
| Path-question repository | Not applicable to HyperGraphRAG |

The selected historical run is `full1000_gpt4omini_20261022_overflow1`. A similarly named `overflowfallback1` directory belongs to a different run and is not the source of the results published here.

## SHGraphRAG + DeepSeek

Each dataset has a separate frozen implementation under [`shgraphrag_deepseek/`](../code/shgraphrag_deepseek/).

| Setting | Value |
| --- | --- |
| Extraction / QA | Supplied tuple codec and original QA prompts |
| Sampling | Seed 42, sorted unique IDs, RNG draw order |
| Input to `DatasetProcessor` | Full development Parquet, sampled exactly once |
| Entity namespace | Question-specific identifiers |
| Construction | Upstream build with documented scheduling, checkpoint, compatibility, and index-readiness repairs |
| Retrieval | Global top 10 vector candidates, then question-ID filter |
| QA top-k / threshold | 3 / 0.3 |
| Similar-entity default | 0.9 in source settings; call-site overrides remain authoritative |
| Path lookup | Online two-hop lookup |
| Offline path-question repository | Not implemented; 0 entries and 0 bytes |
| Generation settings | Temperature, max tokens, and thinking settings omitted |
| Text timeout | 600 seconds |
| Hotpot / MuSiQue scheduling | Dispatch 32; physical request gate 64 |
| 2Wiki final scheduling | Dispatch 100; physical request gate 100 |
| Embeddings | Hotpot/MuSiQue: 4 in flight × up to 32 texts; 2Wiki: 8 × up to 32 |
| Vector indexes | Neo4j entity and passage indexes; readiness checked before QA |
| Storage | Isolated Neo4j + local checkpoints / request artifacts |

The 2Wiki execution history contains a 64-to-100 request-capacity transition. That scheduling compatibility is retained without importing any private result caches. It is inaccurate to label all datasets uniformly `c100` or to infer concurrency-normalized timings.

## HyperGraphRAG + DeepSeek

Authoritative per-dataset sources are under [`hypergraphrag_deepseek/`](../code/hypergraphrag_deepseek/).

| Setting | Value |
| --- | --- |
| Chunk / overlap / gleaning | 1,200 / 100 / 2 |
| Retrieval mode / top-k | Hybrid / 60 |
| Context sections | 4,000 tokens |
| Text and embedding physical gate | Shared ceiling 100; embeddings at most 4 |
| Embedding batch | 32 |
| Embedding request | Local port 11434; `num_batch=8192`; unspecified truncate is retained |
| Visible summary target | At most 500 `o200k_base` tokens |
| Physical summary generation budget | 8,192 |
| Summary length completion | Accept nonempty length-limited output; clip visible prefix to 500 tokens |
| Unavailable summary | Preserve available original description as implemented |
| Summary audit | Store original/adopted text identities, usage, finish reason, and clipping decision |
| General generation parameters | Empty object; no invented temperature / thinking defaults |
| Storage / inference scope | NetworkX / NanoVectorDB / JSON KV; pooled candidate corpus |
| Cache imports in public fresh run | None |
| Empty normal answers | Retained in the 1,000-question denominator |

The summary adaptation is part of the evaluated implementation. The DeepSeek Hyper code is not an untouched upstream baseline. The original working-tree algorithm diff is preserved in each dataset's `provenance/algorithm.patch`.

## Prompt sources

| Profile | Exact prompt files |
| --- | --- |
| SH GPT | [`config/prompt.py`](../code/shgraphrag_gpt4omini/config/prompt.py), [`reproduction/prompts.py`](../code/shgraphrag_gpt4omini/reproduction/prompts.py), QA schema/calls in `reproduction/runner.py` |
| Hyper GPT | [`hypergraphrag/prompt.py`](../code/hypergraphrag_gpt4omini/hypergraphrag/prompt.py), [`revision_openrouter/run.py`](../code/hypergraphrag_gpt4omini/revision_openrouter/run.py) |
| SH DeepSeek | Per-dataset `config/prompt.py`, `qa/prediction.py`, `build/build_graph.py`, and graph extraction modules |
| Hyper DeepSeek | Per-dataset `hypergraphrag/prompt.py`, `experiment/run.py`, `experiment/transport.py` |

Source prompts are published in full. They are preserved in their original language and wording; the documentation does not silently substitute a translated or shortened prompt.

## Public runtime variables

| Variable | Meaning |
| --- | --- |
| `SHGRAPHRAG_DATA_ROOT` | Prepared-data root, set by the wrapper |
| `SHGRAPH_FULL_DEV_PARQUET` | Full normalized dev input for DeepSeek SH |
| `SHGRAPH_BENCHMARK_CONFIG` | Data-only benchmark contract for DeepSeek SH Hotpot/2Wiki |
| `OPENROUTER_API_KEY` | Reader's own GPT credential, inspected only for authorized execution |
| `DEEPSEEK_API_KEY` | Reader's own credential for the configured DeepSeek route |
| `OPENROUTER_PROXY` | Optional process-scoped text proxy; direct by default |
| `TIKTOKEN_CACHE_DIR` | Public tokenizer dependency cache |
| `JAVA_HOME` | Explicit Java 17 installation for DeepSeek SH |
| `SHGRAPHRAG_ALLOW_MODEL_API` | Internal child-process execution opt-in; wrapper sets it for `run` only |

Provider changes are new experimental conditions. Preserve the old record and report the new endpoint/model explicitly instead of calling it the same historical run.
