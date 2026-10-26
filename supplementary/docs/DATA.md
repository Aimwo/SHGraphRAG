# Data, sampling, and corpus reconstruction

[Home](../../README.md) · [Quick start](QUICKSTART.md) · [Evaluation](EVALUATION.md)

The repository ships protocol metadata only. Readers obtain the official releases under their respective terms and reconstruct the inputs locally. All experiments use development questions and their associated candidate passages. They are not searches over full Wikipedia.

## Official inputs

Place these files under `--source-root`:

```text
official-inputs/
└── dataset/
    ├── hotpot_qa/distractor/validation-00000-of-00001.parquet
    └── raw_official/
        ├── 2wiki/release_20210407/
        │   ├── dev.json
        │   └── id_aliases.json
        └── musique/release_v1/data/musique_ans_v1.0_dev.jsonl
```

| Dataset | Official source | Version / split | Development questions |
| --- | --- | --- | ---: |
| HotpotQA | [HotpotQA dataset](https://hotpotqa.github.io/), [pinned Parquet release](https://huggingface.co/datasets/hotpotqa/hotpot_qa/tree/1908d6afbbead072334abe2965f91bd2709910ab/distractor) | Distractor validation, HF commit `1908d6afbbead072334abe2965f91bd2709910ab` | 7,405 |
| 2WikiMultiHopQA | [Official dataset repository](https://github.com/Alab-NII/2wikimultihop) | Official 2021-04-07 release, dev and answer-ID aliases | 12,576 |
| MuSiQue | [Official dataset repository](https://github.com/StonyBrookNLP/musique) | MuSiQue-Ans v1.0 development | 2,417 |

`id_aliases.json` is JSON Lines despite its extension. MuSiQue-Ans is the answerable set; do not substitute MuSiQue-Full or deduplicate its base-ID variants. The Hotpot fullwiki split is not a substitute for distractor validation.

## Release hashes

SHA‑256 is checked before conversion:

| File | SHA‑256 |
| --- | --- |
| Hotpot distractor validation Parquet | `c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6` |
| 2Wiki official `dev.json` | `79f77ae104088ea8e25b1a65dbece768d45771194663bc5660ec9a98070dadf5` |
| 2Wiki official `id_aliases.json` | `f08ffcb6c2cefca9bdbe86b4248d6ad7a7743762d3f7264c14ff0bae85726fb6` |
| MuSiQue-Ans v1.0 dev JSONL | `15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b` |

Do not bypass a mismatch. A different data release can change candidate passages even when the question IDs and answers appear unchanged.

## Frozen samples

For each dataset:

```python
sample_ids = random.Random(seed).sample(sorted(unique_question_ids), 1000)
```

Preserve the RNG draw order through construction, query execution, and evaluation. The six reference lists are under [`protocols/`](../protocols/). The original records also store source and artifact hashes.

| Protocol | Seed | Hotpot passages | 2Wiki passages | MuSiQue passages |
| --- | ---: | ---: | ---: | ---: |
| GPT‑4o mini, both methods | 20261020 | 9,800 | 6,293 | 11,524 |
| DeepSeek, both methods | 42 | 9,773 | 6,361 | 11,835 |

These counts refer to unique candidate passages after exact deduplication. Repeated occurrences retain their question memberships. Matched inputs do not imply identical retrieval scope: DeepSeek SHGraphRAG filters candidates by question, while DeepSeek HyperGraphRAG searches the pooled union.

## Reconstruct

```bash
python supplementary/tools/prepare_data.py \
  --source-root /absolute/path/to/official-inputs \
  --data-root /absolute/path/to/prepared-data
```

The script reads official files and writes only to the separate output root. It performs no network requests or model calls. It:

1. Checks the four raw release hashes, split sizes, and unique native IDs.
2. Preserves questions, answers, candidate paragraph content, and candidate order in normalized tables.
3. Regenerates both 1,000-question samples and compares their complete ordered IDs.
4. Reconstructs candidate corpora and per-question passage mappings.
5. Checks the text artifact hashes against the accepted run manifests.
6. Writes per-profile input manifests and a preparation validation receipt.

The legacy preparation script is preserved as [`original_prepare_paper_datasets.py`](../tools/original_prepare_paper_datasets.py) for provenance. The public helper reuses its conversion functions without invoking the legacy workspace's input-repair behavior.

## Passage identities

The GPT protocol keeps title and text separate:

```text
text = "\n".join(sentences)
passage_id = "passage_" + SHA256(title + "\0" + text)
corpus order = sorted passage IDs
```

The DeepSeek protocol uses the source pipeline's title-prefixed chunk:

```text
text = title + "\n" + "\n".join(sentences)
passage_id = SHA256(text)
corpus order = first occurrence in frozen question order
```

Do not exchange corpus files between these protocols. For MuSiQue, each original paragraph is kept as one sentence-list element; it is not resegmented.

## Generated output

```text
prepared-data/
├── normalized/{hotpotqa,2wiki,musique}.parquet     # Full dev files
├── dataset/paper_ready/manifest.json
├── dataset/paper_ready/<dataset>/sample_1000_seed20261020/
├── dataset/raw_official/...                       # Required native/alias copies
├── gpt4omini/datasets.lock.json
├── gpt4omini/queries/<dataset>.jsonl
├── deepseek/<dataset>/                            # IDs, queries, corpus, gold, aliases
├── deepseek/shgraphrag/<dataset>/benchmark.json    # Hotpot/2Wiki data-only path contract
└── preparation_validation.json
```

Native gold and supporting annotations are evaluation inputs. Inference query files contain exactly `id` and `question`; they exclude answers and supporting facts. Candidate memberships used by construction/retrieval are documented per implementation.

Generated Parquet encoding can differ across PyArrow versions. The public helper records fresh byte hashes while checking native content, sample IDs, text corpus hashes, and mappings against the historical protocol. To minimize such differences, use the pinned preparation environment, Python 3.11 and PyArrow 19.0.1.
