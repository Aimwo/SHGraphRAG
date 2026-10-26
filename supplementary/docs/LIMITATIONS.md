# Interpretation and reproducibility boundaries

[Home](../../README.md) · [Configuration](CONFIGURATION.md) · [Results](RESULTS.md)

## Alternative-backbone revision experiments

The original submitted paper used a Gemini-era setup. This branch publishes completed GPT‑4o mini and DeepSeek revision runs. The new rows must not be relabeled as reproductions of the original Gemini scores.

The two model families use different seeds. Within a family, question IDs and candidate corpora are matched across the two methods. Across families, seed20261020 and seed42 produce different samples and corpus sizes.

## SHGraphRAG implementation gap

The GPT implementation performs global identity reconciliation and builds the full offline two-hop path-question repository. The evaluated DeepSeek implementation preserves the supplied tuple-based QA logic, resolves two-hop paths online, and has no offline path-question index.

Publishing both source variants makes this difference inspectable. It does not resolve the manuscript's implementation-description gap or establish that the two implementations execute the same mechanism.

## Retrieval scope

DeepSeek SHGraphRAG uses question-scoped entity identifiers and filters global top-10 candidates by question ID. DeepSeek HyperGraphRAG retrieves from the pooled candidate union. Shared questions/corpora therefore do not constitute a retrieval-scope-controlled comparison.

All runs use the sampled questions' associated candidate passages. These results do not establish full open-domain scalability over complete Wikipedia.

## Provider and generation defaults

The recorded DeepSeek service is a third-party relay at `https://new.21r.cn/v1`, not an official DeepSeek endpoint. Its future model availability, provider-side defaults, pricing, and behavior are not guaranteed by this code release.

DeepSeek temperature, output-limit, and thinking/reasoning fields were generally omitted. Omission is not evidence that a control was zero, disabled, or identical to another provider's default. Reported zero reasoning tokens also cannot establish that no reasoning content was produced.

GPT uses explicit temperature 0, but neither profile has an established deterministic hosted-model seed. A fresh run can change graph extractions, alias merges, paths, answers, and scores despite identical input IDs.

## Baseline adaptations

The GPT Hyper profile contains audited terminal-outcome handling and an explicit overflow-only embedding truncation fallback. The DeepSeek Hyper profile contains the documented visible-summary policy with an 8,192-token physical generation budget. Neither should be described as an untouched upstream execution.

## Timing, storage, and monetary cost

Historical campaigns included retries, caching, resumes, scheduling transitions, and same-provider cache recovery. Observed query latency can include queueing, cache effects, and shared-host contention.

No complete independent one-pass construction time is claimed for either method. No independent HyperGraphRAG index-creation time is claimed. Dividing wall time by workers or using the last resumed invocation does not supply those missing measurements.

Source records report request/token/storage observations where available. Token usage is not a settled monetary charge; relay pricing was not independently verified. Do not infer cost superiority from unverified tariffs.

## What this publication validates

Publication-time checks reconstruct the official inputs, verify all selected sample/corpus/gold/membership identities, exercise the preparation handoffs, run applicable mock regression suites with service connections blocked, and check release files and documentation.

They do not rerun paid model inference, rebuild the million-row path repository, or reconnect to an existing graph/embedding service. Historical completion receipts and new publication checks have separate labels. Without bundled data and raw predictions, historical scores cannot be independently recomputed from this branch alone.

The supplied pipeline is intended for an independently obtained dataset and a reader-authorized fresh run. Provider substitution, changed corpus scope, different inference defaults, new prompt wording, or a missing optional dependency should be documented as a changed condition rather than hidden by a successful exit code.
