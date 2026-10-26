# Troubleshooting

[Home](../../README.md) · [Quick start](QUICKSTART.md)

| Symptom | Cause and action |
| --- | --- |
| Release hash mismatch | Restore the published source or preserve your edits as a new experiment. Check `.gitattributes` so checkout tools do not change bytes. |
| Official release hash mismatch | Wrong dataset version or a reserialized file. Obtain the exact release in [DATA.md](DATA.md). Do not overwrite the expected hash. |
| Ordered sample IDs differ | Wrong seed, fullwiki instead of distractor, wrong MuSiQue set, duplicate IDs, or a different source release. The helper stops before model calls. |
| DeepSeek SH processor selects another order | A 1,000-row sample was supplied to a processor that samples again. Use the full development Parquet via the wrapper. |
| `SHGRAPHRAG_DATA_ROOT` missing | Use the public planner/wrapper, or explicitly supply the prepared-data root before importing the profile driver. |
| Explicit model opt-in required | Planning/preparation does not authorize inference. For an intentionally authorized run, supply your credential and both `--execute --allow-model-api`. |
| Missing key | Set your own process-scoped `OPENROUTER_API_KEY` or `DEEPSEEK_API_KEY`. No private launcher, `.env`, DPAPI file, or token is bundled. |
| Returned model identity differs | The provider route does not match the recorded allowlist. Stop and record the actual model as a new condition. |
| OpenRouter connection fails | Check the route allowed by your provider. The historical local proxy is optional and process-scoped; never send loopback embedding traffic through it. |
| BGE tag exists but contract fails | Check the complete model digest, 1,024 dimensions, endpoint, batch options, and truncation policy. A tag name alone is insufficient. |
| Tokenizer attempts a network request during offline tests | Warm `o200k_base` or point `TIKTOKEN_CACHE_DIR` to an existing dependency cache before blocking the network. |
| Neo4j port already occupied | The ownership check intentionally rejects another instance. Inspect the service separately; do not redirect the build or terminate an unrelated process. |
| Java path unavailable | Set `JAVA_HOME` to an actual Java 17 installation. The supplied lifecycle helper targets Windows `bin/java.exe`. |
| Dependency installation fails on another platform | Use the recorded Python/runtime combination, examine wheel/build availability, and document any dependency change. Clean reinstalls on every OS were not validated here. |
| Optional adapter has Python syntax/dependency errors | Selected Hyper profiles use NetworkX/JSON/NanoVectorDB. Unused optional adapters may target Python 3.12 or require optional packages; do not silently change the selected storage backend. |
| Hyper GPT refuses a different dataset selection | `execution.json` freezes `--datasets`. Start a full campaign with wrapper `--dataset all`, or use separate run names for independent subsets. |
| Hyper GPT final verification lacks a dataset directory | Complete all three datasets under one all-dataset campaign before invoking its campaign verifier. |
| Reusing a run fails source/config/input checks | Preserve that run, and use a new run name or fresh checkout for changed conditions. Do not delete audit history to bypass the contract. |
| Model returned no answer but the controller completed | Completion and answer nonemptiness differ. Keep the answer in the full denominator and inspect the terminal outcome. |
| Full SH GPT construction is unexpectedly large | Paths are uncapped and MuSiQue had about 2.5 million path-question rows. Plan disk/RAM and budget before executing. |
| A legacy script references an old workstation path | Use the supported wrapper and method guide. Legacy provenance and obsolete provider tests identify historical context, not a public credential dependency. |

When reporting a reproduction issue, include the branch commit, profile, dataset, phase, Python version, non-secret effective config, source/input hashes, and the smallest relevant traceback. Remove keys, authorization headers, private response content, and dataset text before posting publicly.
