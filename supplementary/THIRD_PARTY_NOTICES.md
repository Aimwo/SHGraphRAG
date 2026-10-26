# Third-party source and dependency notices

- **SHGraphRAG**: [Aimwo/SHGraphRAG](https://github.com/Aimwo/SHGraphRAG), base commit `3ed1685fefb1295c43c7fca0bb306e4d5fd08e77`. Original root code is retained under the repository's MIT license. Supplementary runtime repairs and experiment harnesses are recorded separately.
- **HyperGraphRAG**: [LHRLAB/HyperGraphRAG](https://github.com/LHRLAB/HyperGraphRAG), commit `d587cdf8c3fe2be7719557f845324cb3a321f5e2`. Copyright (c) 2025 Haoran Luo, MIT license. A full license copy accompanies every vendored implementation directory. Evaluated working-tree changes remain in the source and in per-dataset algorithm diffs where captured.
- **Dataset evaluation functions**: Native HotpotQA, 2Wiki, and MuSiQue answer-scoring functions and the supplied evaluation adapters are retained with their source comments. Original public sources are linked from [DATA.md](docs/DATA.md). Preserve upstream notices when redistributing these files.
- **Runtime dependencies**: Ollama/BGE‑M3, Neo4j Community, Graph Data Science, APOC, tiktoken, NumPy, PyArrow, LangChain, NanoVectorDB, and the remaining pinned packages retain their own licenses. Dependency locks list versions; they do not relicense the dependencies.
- **Datasets and weights**: No benchmark texts, answers, model weights, or embedding vectors are distributed here. Obtain them from their official sources and follow their terms.

The supplementary diagrams are generated locally from editable source in `assets/`; they contain no external artwork or benchmark examples.
