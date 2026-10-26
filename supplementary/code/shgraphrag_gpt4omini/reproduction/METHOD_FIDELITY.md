# SHGraphRAG implementation audit — 2026-10-21

This is the acceptance policy for `source_alias_merge_global_graph_v2`.
It distinguishes an adapted implementation from a claim to recover the
historical Gemini experiment or its undocumented original question sample.

| Component | Supplied source / manuscript | Isolated implementation |
|---|---|---|
| Extraction | `config/prompt.py`, `build/build_graph.py` | Same entity/relation objective and fields; strict Chat Completions JSON replaces the inconsistent tuple codec. After three structurally invalid outputs, missing endpoint records can be completed under a schema with their exact names as required keys; original accepted entities/relations remain unchanged. This bounded format adaptation uses passage text only and all calls remain counted. |
| Entity descriptions | Extraction prompt requests attributes/activities. Active build/index path contains no separate entity-summary LLM call. | All extracted descriptions survive alias merging and graph assembly. No invented summary stage or summary-cost claim. |
| Alias candidates | `GraphBuilder._merge_local_entities`: name cosine > 0.95 OR substring; within a document's candidate chunks. | Same threshold and OR rule, scoped to the supplied unlabelled candidate groups. Each unique pair is confirmed once. |
| Alias decisions | Same-real-world-entity LLM decision; batches of 10; prefer more frequent canonical name. | OpenRouter `openai/gpt-4o-mini`, temperature 0, no reasoning parameter, strict boolean JSON and local IDs; complete decisions required. Transitive positive decisions reconciled across overlapping groups, canonical name selected by corpus passage frequency with deterministic ties. |
| Entity identity | Historical benchmark code appends `__question_id`, then filters retrieval by that suffix. | One global union graph, no question suffix and no membership filter. Local candidate grouping is a construction input only. This is a substantive scope correction, disclosed in every result report. |
| Sparsification | `_prune_triples_in_memory_2hop`; per original chunk, retain bridge-incident edges and edges incident to their neighbors. | Same rule after confirmed alias remapping; bridge membership counts distinct source hyperedges. Full document text and entity membership survive in hyperedges. |
| Paths | Manuscript asks for all two-hop paths and multiple generated questions. Supplied active QA lacks a complete reusable path repository. | All simple undirected two-edge paths, reversal deduplicated, relation directions preserved; two endpoint-answer questions per path. No degree/count cap. |
| Entity vectors | `EntityIndexManager.create_entity_index(text_properties=['id'])`; strips `__...`. | Canonical name only, BGE-M3; no first-description shortcut. All descriptions remain in the graph. |
| Direct retrieval | Manuscript Top-3 paths and sufficiency gate. | Top-3 distinct paths by their best generated-question cosine; GPT-4o-mini sufficiency and minimal answer. Up to three associated full passages accompany the paths, explicitly disclosed. |
| Decomposition | Minimal one-hop questions, entities, dependency flag; background document retrieval. | Same fields; top-2 global hyperedges supply background; maximum six subquestions is the recorded implementation setting. |
| Progressive retrieval | Manuscript Top-2 entities, connected one-hop triples, Top-3 associated hyperedges; fallback Top-2 hyperedges. | Named-entity BGE linking, up to two highest-scoring entities above the configured 0.3 heuristic inherited from `SIMILARITY_THRESHOLD`; all incident triples and Top-3 associated complete passages. Top-2 global fallback when no association is available. This threshold is not tuned on gold. |
| Dependencies | Independent queries parallel; dependent queries rewritten from previous answers. | Parallel waves of independent subquestions; ordered history and dependent rewriting. All calls share the same worker semaphore. |
| Answers / scoring | Short-answer EM/F1; development sample. Historical scripts have ID and denominator defects. | Gold excluded from inference; exact frozen ID alignment, full denominator, missing predictions zero. Single-reference and alias-aware metrics reported separately. |
| Storage | Original Neo4j builder clears its database without a restriction. | Isolated JSON/SQLite/NumPy exact-cosine storage; never touches an existing Neo4j database. Storage and latency cannot be compared as identical implementations. |

This branch restarts the model experiment on 2026-10-21. Require fresh
`pilot3_4omini_v2_20261021` verification before dispatching
`full1000_4omini_v2_20261021`. Historical Luna outputs remain in their original
worktree and are not consumed by this run. All earlier pilot summaries under
`results/20261021/` are historical, not GPT-4o-mini measurements.

The v2 verifier independently checks every alias decision, reconstructs the
canonical mapping and graph from saved extractions, checks all generated paths,
index dimensions and normalization, exact prediction IDs, model/effort usage,
implementation hashes and the evaluation denominator.

## Corpus-scope evidence

The working manuscript's Datasets paragraph
(`Paper_SHGraphRAG/Edit/Paper/manuscript.tex`) explicitly constructs one retrieval
corpus **for each dataset** from all supporting and distractor passages
associated with the 1,000 selected questions. The 2026-10-15 reproduction audit
also warns that the supplied code's question-ID-restricted context retrieval is
not interchangeable with that dataset-level corpus.

This reproduction follows the manuscript's dataset-level candidate union.
Native per-question distractor contexts are a legitimate *different benchmark
setting*; they are not intrinsically answer-label leakage. The historical
implementation is disclosed as using that different scope, rather than being
silently substituted for a global retrieval run. No full open-domain Wikipedia
claim is made for either setting.

The MuSiQue v2 pilot independently enumerates 3,732 paths from its 60-passage
union. Five require edges drawn across native candidate groups. Its saved
`construction_scope_audit.json` records this check. The formal runs keep the
same scope and do not reduce paths to native question membership.
