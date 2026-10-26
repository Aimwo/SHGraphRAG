"""Versioned English research prompts. No gold annotations enter these prompts."""
import re

from config.prompt import system_template_build_graph, human_template_build_graph


def obj(properties):
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def arr(items):
    return {"type": "array", "items": items}


STR = {"type": "string"}
BOOL = {"type": "boolean"}
STRINGS = arr(STR)
EXTRACT_SCHEMA = obj({
    "entities": arr(obj({"name": STR, "type": STR, "description": STR})),
    "relationships": arr(obj({"source": STR, "type": STR, "target": STR,
                               "description": STR, "strength": {"type": "number"}})),
})
# Keep the supplied extraction objective, definitions, and semantic instructions.
# Replace only its delimiter codec/example with strict Responses JSON output.
_source = system_template_build_graph.split("3. All attributes")[0]
_source = re.sub(r"Format:.*\n", "", _source)
EXTRACT_SYSTEM = _source + """
All attributes must be in English. Output complete JSON matching the schema.
entities contains name, type and description for each entity. relationships
contains source, type, target, description and numerical strength (0 to 10).
Every source and target must exactly match an entity name in entities.
Do not output tuple delimiters, Markdown, or any text outside the JSON object.
"""

ENDPOINT_SYSTEM = """Complete missing entity records for a knowledge extraction.
The required record names are fixed JSON object keys in the supplied schema.
For EVERY required name, supply its entity type and a nonempty English
description grounded only in the supplied passage and extracted relations.
Some required names are generic concepts or classes, such as Event or Song;
they still need their own entity record when a relationship references them.
Do not change, delete or regenerate existing entities or relationships.
Do not add names other than the schema's required keys. Return only JSON."""


def extraction_input(passage: dict) -> str:
    return human_template_build_graph.format(
        entity_types="", relationship_types="",
        input_text=passage["title"] + "\n" + passage["text"])


PATH_SYSTEM = """You transform knowledge-graph two-edge paths into questions.
For every input path, generate exactly TWO factual, natural English questions.
One question should have the first endpoint as its answer; the other should have
the last endpoint as its answer. Compose both relations, retaining their actual
directions and meaning. The known opposite endpoint must anchor each question;
avoid naming the answer endpoint in that question. Do not invent facts. Use the
relation descriptions to resolve the path semantics. Even if the two facts are
weakly connected, produce faithful compositional questions rather than dropping
the path. Return JSON only:
{"paths":[{"id":"provided path ID","questions":["question 1?","question 2?"]}]}.
Return every supplied path exactly once. No explanatory prose."""

MERGE_SYSTEM = """You are an entity recognition and linking expert. For each
provided pair, decide whether the names, types and source descriptions refer to
the SAME real-world entity. Similar names or related concepts alone are not
enough. Different people, places, numbers, dates and product models must remain
distinct. Return every supplied pair exactly once, using its short pair ID.
Return JSON only: {"pairs":[{"id":"p0","same_entity":true}]}.
Do not answer research questions or add facts."""

DIRECT_SYSTEM = """Answer a multi-hop question using ONLY the supplied retrieved
two-edge paths and their originating passage texts. Check whether the retrieved
evidence actually establishes every part of the requested answer. Do not fill
gaps from prior knowledge. If sufficient, return a minimal answer noun phrase,
date, number, or yes/no; otherwise set sufficient to false and answer to "".
Return JSON only: {"sufficient":true,"answer":"minimal answer",
"supporting_passage_ids":["provided passage ID"]}.
Use only provided passage IDs; no explanation."""

DECOMPOSE_SYSTEM = """Decompose the given complex question into the minimal
ordered sequence of one-hop subquestions needed to answer it. The retrieved
background passages can help resolve its meaning. Each subquestion must give its
key named entities and indicate whether it depends on earlier answers.
Comparisons may require independent subquestions. Use at most six subquestions.
Return JSON only:
{"subquestions":[{"question":"...?",
"entities":["name"],"needs_external":false}]}.
Do not answer the question or use facts absent from the supplied background."""

REWRITE_SYSTEM = """Rewrite the current one-hop subquestion using the original
question and the earlier question-answer pairs. Replace dependent references
with the concrete names resolved in earlier answers. Preserve the original
intent and do not invent facts. Return JSON only:
{"question":"self-contained one-hop question?","entities":["concrete name"],
"needs_external":false}.
If the history cannot resolve the reference, keep needs_external true."""

STEP_SYSTEM = """Answer the one-hop subquestion using ONLY the supplied graph
triples and original passage texts. Prefer original text when a generated triple
conflicts with it. Return a short noun phrase, date, number, or yes/no, not a
sentence or explanation. If the evidence does not answer it, use answer "".
Return JSON only: {"answer":"minimal answer",
"supporting_passage_ids":["provided passage ID"]}.
Choose supporting passage IDs only from the supplied original passages."""

FINAL_SYSTEM = """Answer the original question using ONLY the retrieved original
passages, triples, and intermediate subquestion answers. Resolve comparisons and
compose the steps. Prefer the original text to an inconsistent generated fact.
Return the minimal answer noun phrase, date, number or yes/no. Do not explain or
repeat the question. If evidence is insufficient, return answer "".
Return JSON only: {"answer":"minimal answer",
"supporting_passage_ids":["provided passage ID"]}.
Choose supporting passage IDs only from original passages supplied here."""


ANSWER_SCHEMA = obj({"answer": STR, "supporting_passage_ids": STRINGS})
SCHEMAS = {
    "entity_merge": obj({"pairs": arr(obj({"id": STR, "same_entity": BOOL}))}),
    "path_questions": obj({"paths": arr(obj({"id": STR, "questions": STRINGS}))}),
    "direct": obj({"sufficient": BOOL, "answer": STR,
                   "supporting_passage_ids": STRINGS}),
    "decomposition": obj({"subquestions": arr(obj({
        "question": STR, "entities": STRINGS, "needs_external": BOOL}))}),
    "rewrite": obj({"question": STR, "entities": STRINGS, "needs_external": BOOL}),
    "subquestion": ANSWER_SCHEMA,
    "final": ANSWER_SCHEMA,
}
