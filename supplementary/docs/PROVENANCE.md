# Source provenance and publication audit

[Home](../../README.md) · [Source manifest](../provenance/source_manifest.json) · [Portability diff](../provenance/portability.patch)

Sources and successful-run evidence were audited on October 26–27, 2026. The publication receipt records its validation timestamp separately from the experiment timestamps. This branch is an additional reproducibility package; the original root implementation remains in place.

## Source selection

| Profile | Source of the exported implementation | Evidence |
| --- | --- | --- |
| SH GPT | The isolated GPT working copy, after all 15 core module hashes were checked against the accepted three-dataset `code_manifest.json` | [`records/shgraphrag_gpt4omini/`](../records/shgraphrag_gpt4omini/) |
| Hyper GPT | `source_snapshot/` of `full1000_gpt4omini_20261022_overflow1`, plus explicitly identified offline acceptance helpers | [`records/hypergraphrag_gpt4omini/`](../records/hypergraphrag_gpt4omini/) |
| SH DeepSeek | Each completed run's immutable source ZIP, checked file-by-file against its execution source manifest | [`records/shgraphrag_deepseek/`](../records/shgraphrag_deepseek/) |
| Hyper DeepSeek | Each final accepted per-dataset run's `source_snapshot/`, original algorithm patch, and dependency lock | [`records/hypergraphrag_deepseek/`](../records/hypergraphrag_deepseek/) |

Base SHGraphRAG commit: `3ed1685fefb1295c43c7fca0bb306e4d5fd08e77`. HyperGraphRAG upstream commit: `d587cdf8c3fe2be7719557f845324cb3a321f5e2`. A Git commit alone does not identify the evaluated working tree: the exported run snapshots include its applicable pre-existing modifications.

The source manifest stores the original source location and SHA‑256, published SHA‑256, size, and whether a publication change occurred. It also covers new public helpers and documentation. The exported source's checksum is not its old upstream commit hash.

## Publication portability changes

All exported-source edits are visible in [`portability.patch`](../provenance/portability.patch). They are confined to:

- Replacing private DPAPI/key-file readers and old workstation credential launchers with explicit reader-supplied environment credentials and a fresh-run opt-in.
- Relocating prepared data, benchmark contracts, Java paths, and dependency-lock references; preserving full-development input to the original seed42 sampler.
- Capturing source from the vendored implementation instead of relying on an outer/private Git layout.
- Checking the frozen algorithm file hashes instead of requiring the new publication commit to equal an upstream commit.
- Supporting optional process-scoped GPT text proxy routing while keeping embedding traffic direct.
- Removing a private campaign-wide pause dependency; fresh DeepSeek Hyper runs use their own opt-in and local stop marker.
- Starting fresh runs without private cache/history imports, and handling the resulting empty imported-timing history explicitly.
- Relocating offline test references to the published native evaluation scripts and preserving the 2Wiki capacity-transition helper.
- Respecting an explicit tokenizer dependency cache, without changing tokenizer identity.
- Sanitizing operator-specific provenance fields and omitting per-example data from exported metadata.

These edits do not change the captured graph/extraction/retrieval algorithms or prompt wording. Frozen HyperGraphRAG algorithm modules and prompt files retain their original hashes; the publication audit records those checks. New shared scripts orchestrate the existing stages without choosing new scientific hyperparameters.

## Excluded material

This package excludes benchmark question/passage text, native answers, predictions, per-example metrics, graph and vector stores, model responses, caches, weights, environments, private credentials, and unrelated project methods. Ordered IDs, hashes, aggregate metrics, and source configurations are retained as reproducibility metadata.

Large per-attempt embedding-usage rows are reduced to counts and native-field totals. Internal call-ID lists and individual failure-attempt records are omitted with their counts retained. These are publication redactions of metadata; the original source-file hashes continue to identify the complete local evidence.

Historical superseded attempts, pilot runs, and empty-answer retry supplements are not alternative table rows. Their existence is described where it affects cache/timing interpretation. The original complete evidence remains in the research workspace; this public export does not bundle those data artifacts.

Legacy source files can still name historical providers, local paths, or obsolete tests. They are retained only where needed to explain the source tree. Supported entry points do not read those old credential stores. In particular, the GPT Hyper inherited legacy transport constructor is disabled, and the DeepSeek test runner excludes the retired-provider `test_runtime.py`.

## Publication validation

[`release_validation.json`](../provenance/release_validation.json) records:

- Official release hashes and reconstruction of both sampling protocols.
- Exact ordered IDs and historical corpus/gold/membership text hashes.
- Preparation of all 12 profile/dataset combinations and real seed42 processor alignment.
- Applicable regression suites across the four profiles, including each dataset-specific DeepSeek source variant.
- Mock model/embedding transports with external/service socket connections blocked.
- Source and prompt identity checks, release file checksums, JSON parsing, documentation links, excluded-file inspection, and literal-byte preservation.

Publication checks do not contact the historical relay, OpenRouter, Ollama, or an existing Neo4j service. They do not establish a fresh 12,000-question inference run or a clean dependency installation on every platform. The historical acceptance receipts and this offline release receipt answer different questions.

## Documentation conventions

The landing page follows the navigation and onboarding structure described in GitHub's [README guidance](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes). Configuration, tutorials, explanations, and evidence have separate pages. Relative links and locally stored illustrations keep navigation tied to this branch.

The method guide uses GitHub-supported [Mermaid diagrams](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/creating-diagrams). The two larger overview figures are editable SVGs generated locally by [`generate_diagrams.py`](../assets/generate_diagrams.py); optional PNG previews are included. No external image-generation service is required.

Both landing pages use a centered title, linked badges, compact documentation navigation, and a language switch. The five static badges describe the license, recorded Python versions, four experiment profiles, three benchmarks, and BGE‑M3 embedding. They link to the corresponding local documentation; they do not report CI results, a package release version, or cross-platform validation. Regenerate them offline with [`generate_badges.py`](../assets/generate_badges.py). The release validator checks the Markdown links and the HTML navigation/image paths used by the mastheads.
