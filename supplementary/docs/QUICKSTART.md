# Reproduce a completed experiment

[Home](../../README.md) · [Environment reference](ENVIRONMENT.md) · [Data protocol](DATA.md)

This tutorial starts from a fresh checkout and reader-supplied official development files. Run commands from the repository root unless a step explicitly changes directory. Use **one virtual environment per profile**. The recorded environments were Windows x86‑64; the SHGraphRAG DeepSeek database launcher specifically targets Windows and Java 17.

## 1. Check out the supplementary branch

```bash
git clone --single-branch --branch Sumpplementary https://github.com/Aimwo/SHGraphRAG.git
cd SHGraphRAG
git branch --show-current
python supplementary/tools/validate_release.py
```

Expected branch: `Sumpplementary`. Release validation uses the standard library and checks published file hashes, JSON, relative documentation links, protocol ID counts, and excluded-file rules. It does not download datasets or contact a model.

## 2. Install a profile environment

Example for SHGraphRAG + GPT‑4o mini on Windows:

```powershell
py -3.11 -m venv supplementary/.venv-sh-gpt
& supplementary/.venv-sh-gpt/Scripts/python.exe -m pip install -r supplementary/code/shgraphrag_gpt4omini/reproduction/requirements.lock.txt
$python = (Resolve-Path supplementary/.venv-sh-gpt/Scripts/python.exe).Path
```

For other profiles, use the interpreter version and requirements path in [ENVIRONMENT.md](ENVIRONMENT.md). Do not install all four environments into one environment. The original root `requirements.txt` is not the lock file for the supplementary profiles.

On Bash, the equivalent environment executable is `supplementary/.venv-sh-gpt/bin/python`. The pure Python profiles can be invoked there, but this release does not claim a fresh Linux end-to-end validation.

HyperGraphRAG uses `o200k_base` from `tiktoken`. In its environment, prepare the public tokenizer dependency before an offline test:

```powershell
$env:TIKTOKEN_CACHE_DIR = Join-Path $env:TEMP 'data-gym-cache'
& $python -c "import tiktoken; tiktoken.get_encoding('o200k_base'); print('tokenizer ready')"
```

The first warm-up may download a public tokenizer asset. It performs no language-model inference and needs no provider key. See [environment details](ENVIRONMENT.md#tokenizer-dependency) for air-gapped setups.

## 3. Reconstruct both frozen data protocols offline

Obtain the four official input files listed in [DATA.md](DATA.md) and place them under a read-only source root. Use a separate output root:

```powershell
$sourceRoot = 'D:/research/official-inputs'
$dataRoot = 'D:/research/shgraphrag-prepared'
& $python supplementary/tools/prepare_data.py --source-root $sourceRoot --data-root $dataRoot
```

This command reads local files only. It verifies official release hashes, regenerates both seeds, compares the ordered IDs to the six published ID lists, and checks the reconstructed corpus, native gold, and membership hashes against the accepted campaigns. It creates a `preparation_validation.json` receipt. Generated Parquet bytes are recorded for the reader's installed PyArrow version; scientific equivalence also checks the official native fields and context order.

Keep `$dataRoot` out of Git. No full Wikipedia corpus is needed for these sampled-candidate experiments.

## 4. Prepare the profile handoff

Example:

```powershell
& $python supplementary/tools/reproduce.py --profile sh-gpt4omini --dataset hotpotqa --data-root $dataRoot --run-name reproduce_01 --phase prepare --execute
```

Use the same `--run-name`, `--workers`, dataset, and environment for preparation and execution. SH GPT defaults to 64 workers; Hyper GPT defaults to 32. Explicit `--workers` must be identical across phases when a run contract is already saved.

For DeepSeek SHGraphRAG, preparation freezes the sample and executes the real `DatasetProcessor` offline, checking all 1,000 IDs and candidate texts. Supply the **full development Parquet**, as the wrapper does; passing the 1,000-row sample into that processor would apply sampling twice.

For DeepSeek HyperGraphRAG, preparation copies and verifies the reconstructed inputs into the selected per-dataset implementation. Its readiness check reports model calls as paused until execution is explicitly enabled; this is expected during offline preparation.

## 5. Provide embedding and graph services

All profiles require a local Ollama BGE‑M3 installation with the recorded model digest. The historical endpoints are:

| Profile | Local embedding base | Extra service |
| --- | --- | --- |
| SH GPT | `http://127.0.0.1:11434` | None; JSON / SQLite / NumPy storage |
| Hyper GPT | `http://127.0.0.1:11436` | None; NetworkX / NanoVectorDB / JSON KV |
| SH DeepSeek | `http://127.0.0.1:11434` | Private Neo4j 5.26.0 + GDS 2.13.4 + APOC 5.26.0 |
| Hyper DeepSeek | `http://127.0.0.1:11434` | None; NetworkX / NanoVectorDB / JSON KV |

See each method guide for exact service setup. The SH DeepSeek construction path clears its graph; the supplied ownership guard permits only its isolated loopback instance. Do not redirect it to an existing database. Do not launch competing jobs against an embedding service without accounting for shared capacity.

## 6. Inspect the execution plan

```powershell
& $python supplementary/tools/reproduce.py --profile sh-gpt4omini --dataset hotpotqa --data-root $dataRoot --run-name reproduce_01
```

The plan prints the working directory, non-secret environment overrides, output path, and exact child commands. It reads no key and starts no model work.

## 7. Execute using your own authorized model access

For GPT profiles, provide `OPENROUTER_API_KEY` through your own process environment or secret manager. For DeepSeek profiles, provide `DEEPSEEK_API_KEY` for the explicitly configured route. No `.env` file or workstation credential launcher is required. Do not paste secrets into source code or a committed command.

The GPT publication defaults to a direct external connection. If you intentionally need the historical local HTTP proxy, set only this process:

```powershell
$env:OPENROUTER_PROXY = 'http://127.0.0.1:6789'
$env:NO_PROXY = 'localhost,127.0.0.1,::1'
```

With your own authorized credential already supplied:

```powershell
& $python supplementary/tools/reproduce.py --profile sh-gpt4omini --dataset hotpotqa --data-root $dataRoot --run-name reproduce_01 --phase run --execute --allow-model-api
```

The command is synchronous and writes progress and checkpoints under the implementation directory. Repeating it resumes that exact run contract; changing model, inputs, code, workers, or configuration may be rejected. Preserve the old output and use a new run name for a new experiment. Hyper DeepSeek has one `run/` per dataset, so use a fresh checkout for another independent run.

For a complete Hyper GPT campaign, use `--dataset all`. Its dataset selection is frozen in `execution.json`; changing from one dataset to another under the same run name is rejected. See the [Hyper GPT guide](HYPERGRAPHRAG_GPT4OMINI.md).

## 8. Score and verify saved outputs

```powershell
& $python supplementary/tools/reproduce.py --profile sh-gpt4omini --dataset hotpotqa --data-root $dataRoot --run-name reproduce_01 --phase evaluate --execute
& $python supplementary/tools/reproduce.py --profile sh-gpt4omini --dataset hotpotqa --data-root $dataRoot --run-name reproduce_01 --phase verify --execute
```

Evaluation and verification operate on reader-generated artifacts; historical aggregate metadata alone cannot recreate per-request audits. Hyper GPT's campaign verifier expects all three datasets under the same run name. Run all three before that final verification.

## 9. Run offline regression tests

```powershell
& $python supplementary/tools/offline_tests.py --profile sh-gpt4omini --data-root $dataRoot
```

For a DeepSeek profile, also pass `--dataset hotpotqa`, `2wiki`, or `musique`. Hyper tests use prepared inputs, so run profile preparation first. The test harness blocks connections to external and existing local services; model and embedding fixtures use mocks. Windows asyncio's internal socketpair is allowed. The [publication validation receipt](../provenance/release_validation.json) records what was checked for this export, separately from historical experiment completion.
