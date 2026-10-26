# Environment and service reference

[Home](../../README.md) · [Quick start](QUICKSTART.md) · [Configuration](CONFIGURATION.md)

## Isolated Python environments

| Profile | Recorded Python | Dependency source | Notes |
| --- | --- | --- | --- |
| `sh-gpt4omini` | 3.11.15 | [`reproduction/requirements.lock.txt`](../code/shgraphrag_gpt4omini/reproduction/requirements.lock.txt) | NumPy 1.26.4, PyArrow 19.0.1, requests 2.32.3 |
| `hyper-gpt4omini` | 3.11.15 | [`revision/requirements.lock.txt`](../code/hypergraphrag_gpt4omini/revision/requirements.lock.txt) | Vendored HyperGraphRAG; use selected NetworkX / NanoVectorDB / JSON storage |
| `sh-deepseek` | 3.12.13 | Per dataset [`requirements.runtime.txt`](../code/shgraphrag_deepseek/hotpotqa/reproduction/requirements.runtime.txt) and [`runtime.constraints.txt`](../code/shgraphrag_deepseek/hotpotqa/reproduction/runtime.constraints.txt) | NumPy 1.26.2, PyArrow 16.1.0, datasets 3.4.1, pinned LangChain stack |
| `hyper-deepseek` | 3.11.15 | Per dataset [`requirements.lock.txt`](../code/hypergraphrag_deepseek/hotpotqa/requirements.lock.txt) | Frozen selected runtime, not the original root requirements |

For the DeepSeek SH runtime:

```powershell
py -3.12 -m venv supplementary/.venv-sh-deepseek
& supplementary/.venv-sh-deepseek/Scripts/python.exe -m pip install -r supplementary/code/shgraphrag_deepseek/hotpotqa/reproduction/requirements.runtime.txt -c supplementary/code/shgraphrag_deepseek/hotpotqa/reproduction/runtime.constraints.txt
```

For DeepSeek HyperGraphRAG:

```powershell
py -3.11 -m venv supplementary/.venv-hyper-deepseek
& supplementary/.venv-hyper-deepseek/Scripts/python.exe -m pip install -r supplementary/code/hypergraphrag_deepseek/hotpotqa/requirements.lock.txt
```

Use the matching file under `2wiki/` or `musique/` when selecting that implementation. The locks are historical package snapshots. This export's tests reused the corresponding installed isolated environments; it did not reinstall every package from a clean package index.

Keep UTF‑8 local to the process:

```powershell
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
```

The planner sets `PYTHONHASHSEED=0` for GPT profiles and `42` for DeepSeek profiles in child processes. Hash seeds do not make hosted LLM generation bitwise deterministic.

## BGE‑M3

| Property | Recorded identity |
| --- | --- |
| Ollama model | `bge-m3:latest` |
| Model digest | `7907646426070047a77226ac3e684fbbe8410524f7b4a74d02837e43f2146bab` |
| Output dimension | 1,024 |
| Text batch size | Up to 32 per physical embedding request |
| SH GPT storage | Normalized float32 arrays, exact cosine search |
| Hyper storage | Upstream NanoVectorDB plus native entity/hyperedge/chunk stores |
| Local route | Direct loopback; never through the text-model proxy |

Install Ollama and obtain BGE‑M3 independently. `latest` is a mutable tag, so verify the **digest**, not just the tag. The supplied code checks embedding dimensions and records model identity; the historical contract files remain in `records/`.

The GPT Hyper guide supplies a separate process-scoped Windows launcher for port 11436. Other profiles historically used port 11434. Do not assume a reader's existing service is the right model or configuration.

## Neo4j for SHGraphRAG + DeepSeek

| Component | Version |
| --- | --- |
| Neo4j Community | 5.26.0 |
| Graph Data Science | 2.13.4 |
| APOC Core | 5.26.0 |
| Java | 17, x86‑64 |

The vendored `reproduction.prepare_services` downloader obtains public distributions and records artifact hashes. It defaults to direct downloading. `SHGRAPHRAG_DOWNLOAD_PROXY` can select an explicit process-scoped proxy for the Neo4j distribution download; it is not persisted globally.

The database startup helper checks executable paths, port ownership, process creation time, and the isolated home directory before accepting a connection. Its Windows startup path requires `JAVA_HOME/bin/java.exe`. For another platform, adapt the lifecycle helper in a separate experiment and record that change.

Historical private ports:

| Dataset | Bolt | HTTP |
| --- | ---: | ---: |
| HotpotQA | 21887 | 21474 |
| 2Wiki | 22887 | 22474 |
| MuSiQue | 20887 | 20474 |

Authentication was disabled only on these owned loopback instances. This is not a deployment configuration for a public service. Each instance uses initial heap 1 GB, maximum heap 3 GB, and page cache 1 GB.

## Tokenizer dependency

HyperGraphRAG uses `tiktoken` with `o200k_base`. Prepare its public cache before disconnecting from the network:

```powershell
$env:TIKTOKEN_CACHE_DIR = 'D:/research/tokenizer-cache'
& $python -c "import tiktoken; tiktoken.get_encoding('o200k_base'); print('ready')"
```

For an offline machine, copy an already populated compatible tokenizer cache and point the same environment variable to it. This asset is a software dependency, not a benchmark dataset, a model-response cache, or a provider key. The public runner respects an explicit `TIKTOKEN_CACHE_DIR`. Do not replace the tokenizer or bypass its checks to make a test pass.

## Resource expectations

The DeepSeek SH historical configuration records Windows 11, 24 logical / 24 physical CPU counts, and approximately 63.4 GiB system RAM. These are provenance observations, not a measured minimum.

The SH GPT implementation retains every generated path-question row. The accepted MuSiQue run had 2,499,468 rows: a single 1,024-dimensional float32 matrix at this size is about 9.5 GiB before additional working arrays, indexes, Python structures, and raw artifacts. A fresh full run can therefore require substantial memory, disk, time, and model budget. No storage, weights, or private caches are bundled to hide those costs.

Do not infer a hardware-normalized speed ranking from these records. The implementations use different storage backends, execution scopes, concurrency, and cache histories; [LIMITATIONS.md](LIMITATIONS.md) defines the comparison boundaries.
