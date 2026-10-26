import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from config.neo4jdb import get_db_manager
from config.settings import *
from model.get_models import get_llm_model, get_embeddings_model
from reproduction.runtime import call_context, run_root, write_json, append_jsonl

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HotpotKGQA")
len_hoppath = [0,0]
pre_id = []

# ------------------------------
# Dataclasses for intermediate data
# ------------------------------
@dataclass
class SubQuestion:
    id: int
    ori_question_id: str
    ori_question: str
    question: str
    question_type: str
    entities: List[str] = field(default_factory=list)
    needs_external: bool = False
    answer: Optional[str] = None
    supporting_chunks: List[Dict] = field(default_factory=list)
    solved: bool = False


@dataclass
class QAResult:
    qid: str
    answer: str
    sp: List[Tuple[str, int]] = field(default_factory=list)


# ------------------------------
# Helper functions for Graph access
# ------------------------------
class GraphHelper:
    def __init__(self):
        self.db = get_db_manager()
        # use graph if available else driver
        self.graph = self.db.graph
        self.driver = self.db.get_driver()

    def _query_graph(self, cypher: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        params = params or {}
        # Prefer graph.query if available
        if hasattr(self.graph, "query"):
            try:
                res = self.graph.query(cypher, params=params)
                # normalize result to list-of-dict
                if isinstance(res, pd.DataFrame):
                    return res.to_dict(orient="records")
                return list(res)
            except Exception as e:
                logger.debug("graph.query failed, falling back to driver: %s", e)
        # fallback to driver
        if self.driver:
            try:
                with self.driver.session() as session:
                    result = session.run(cypher, params)
                    return [record.data() for record in result]
            except Exception as e:
                logger.error("Neo4j query failed: %s\nCypher: %s", e, cypher)
                return []
        logger.error("No graph interface available")
        return []

    def find_entities_by_name(self, name: str, fuzzy: bool = True, limit: int = 10) -> List[Dict[str, Any]]:
        """
        在 KG 中按照名称查找可能的实体（exact 或 模糊）
        修改：实体的名称字段是 'id' 而不是 'name'
        """
        # try exact match first
        cypher_exact = f"""
        MATCH (e:{ENTITY_LABEL})
        WHERE toLower(e.id) = toLower($name)
        RETURN e.id AS entity_id, e.description AS description, e.{ENTITY_EMBED_FIELD} AS embedding, 
               e.type AS entity_type, e.title AS title
        LIMIT $limit
        """
        res = self._query_graph(cypher_exact, {"name": name, "limit": limit})
        if res:
            return res
        if fuzzy:
            # substring match
            cypher_fuzzy = f"""
            MATCH (e:{ENTITY_LABEL})
            WHERE toLower(e.id) CONTAINS toLower($name) OR toLower(e.description) CONTAINS toLower($name)
            RETURN e.id AS entity_id, e.description AS description, e.{ENTITY_EMBED_FIELD} AS embedding,
                   e.type AS entity_type, e.title AS title
            LIMIT $limit
            """
            res2 = self._query_graph(cypher_fuzzy, {"name": name, "limit": limit})
            return res2
        return []

    def get_nodes_with_embeddings(self, embedding: List[float], label: str,
                                  limit: int = 10, qid: str = "") -> List[Dict[str, Any]]:
        """
        通用方法：根据节点标签检索相似节点（可用于 __Entity__ 或 __Chunk__）。
        修改：根据不同标签返回不同的属性字段
        """

        if label == ENTITY_LABEL:
            index_name = ENTITY_VECTOR
            cypher = f"""
            CALL db.index.vector.queryNodes($index_name, $limit, $query_embedding)
            YIELD node, score
            MATCH (node:{label})
            WHERE node.embedding IS NOT NULL
            RETURN DISTINCT node.id AS entity_id, node.description AS description, 
                   node.embedding AS embedding, node.type AS entity_type, 
                   node.title AS title, score
            """
        else:  # CHUNK_LABEL
            """
            得到包含这个问题的doc的所有chunk"""
            index_name = CHUNK_VECTOR
            # cypher = f"""
            #         CALL db.index.vector.queryNodes($index_name, 1000, $query_embedding)
            #         YIELD node, score
            #         MATCH (node:{label})--(d:__Document__)
            #         WHERE node.embedding IS NOT NULL
            #           AND d.fileName CONTAINS $qid
            #         RETURN DISTINCT node.text AS text, node.id AS id, score,
            #                node.title AS title, node.fileName AS fileName
            #         """
            cypher = f"""
            MATCH (d:__Document__)
            WHERE d.fileName CONTAINS $qid
            MATCH (d)--(c:{label})
            WHERE c.embedding IS NOT NULL
            RETURN DISTINCT c.text AS text, c.id AS id,
                   c.title AS title, c.fileName AS fileName
            """

        params = {
            "index_name": index_name,
            "limit": limit,
            "query_embedding": np.asarray(embedding, dtype=float).tolist()
        }
        if qid:
            params["qid"] = qid

        return self._query_graph(cypher, params)

    def get_entity_one_hop(self, entity_id: str) -> Dict[str, Any]:
        """
        获取实体的一跳邻居信息
        修改：根据实际的实体属性字段进行查询
        """
        cypher = f"""
        MATCH (e:{ENTITY_LABEL} {{id: $eid}})-[r]-(n)
        RETURN e.id AS entity_id, e.description AS description, e.type AS entity_type,
               collect({{
            rel: type(r),
            props: properties(r),
            neighbor: CASE
                WHEN n:{CHUNK_LABEL} THEN {{
                    label: '{CHUNK_LABEL}', 
                    text: n.text, 
                    id: n.id,
                    length: n.length
                }}
                WHEN n:{ENTITY_LABEL} THEN {{
                    label: '{ENTITY_LABEL}',
                    entity_id: n.id,
                    description: n.description,
                    entity_type: n.type,
                    title: n.title
                }}
                ELSE {{label: labels(n)[0], props: properties(n)}}
            END
        }}) AS neighbors
        LIMIT 1
        """
        res = self._query_graph(cypher, {"eid": entity_id})
        return res[0] if res else {}

    def get_entity_chunks(self, entity_id: str) -> List[Dict[str, Any]]:
        """
        查找提到该实体的所有chunk
        修改：根据实际的chunk属性字段进行查询
        """
        cypher = f"""
        MATCH (c:{CHUNK_LABEL})-[r:MENTIONS]->(e:{ENTITY_LABEL} {{id: $eid}})
        RETURN c.text AS text, c.id AS id, c.title AS title, collect(r) AS relations
        LIMIT 4
        """
        return self._query_graph(cypher, {"eid": entity_id})

    def find_most_similar_chunk(self, chunk_ids: List[str], embedding: List[float]) -> List[Dict[str, Any]]:
        """在给定的一组 chunk 节点中，计算每个节点的向量相似度，
然后返回与查询最相似的前 2 个 chunk。"""
        cypher = f"""
                UNWIND $chunk_ids AS chunk_id
                MATCH (c:{CHUNK_LABEL} {{id: chunk_id}})
                WHERE c.embedding IS NOT NULL
                WITH c, vector.similarity.cosine(c.embedding, $query_embedding) AS score
                ORDER BY score DESC
                LIMIT 2
                RETURN c.text AS text, c.id AS id, score,
                       c.title AS title, c.fileName AS fileName
                """

        params = {
            "chunk_ids": chunk_ids,
            "query_embedding": np.asarray(embedding, dtype=float).tolist()
        }

        result = self._query_graph(cypher, params)

        # _query_graph 预期返回一个列表，因为我们用了 LIMIT 1，
        # 所以列表最多只有一个元素。
        return result if result else []

    def get_2hop_path(self, entity_id):
        """根据实体id获得其2跳路径"""

        cypher = f"""MATCH (e1:__Entity__ {{id: $entity_id}})-[r1]-(e2:__Entity__)-[r2]-(e3:__Entity__)
        WHERE 
          NOT e1:__Chunk__ AND NOT e1:__Document__ AND
          NOT e2:__Chunk__ AND NOT e2:__Document__ AND
          NOT e3:__Chunk__ AND NOT e3:__Document__

        
        RETURN 
            e1.id AS e1, e1.description AS e1_desc,
          type(r1) AS rel1, r1.description AS rel1_desc,
          e2.id AS e2, e2.description AS e2_desc,
          type(r2) AS rel2, r2.description AS rel2_desc,
          e3.id AS e3, e3.description AS e3_desc
        LIMIT 10000000
        """
        return self._query_graph(cypher, {"entity_id": entity_id})
# ------------------------------
# Pipeline core
# ------------------------------
class HotpotKGPipeline:
    def __init__(self,
                 dataset_path: str = "dataset/hotpot_qa/distractor/validation-00000-of-00001.parquet",
                 output_dir: str = "./outputs",
                 top_k: int = TOP_K,
                 sim_threshold: float = SIMILARITY_THRESHOLD,
                 ids_path: Optional[str] = None):
        self.dataset_path = dataset_path
        source_dataset = load_dataset("parquet", data_files=str(self.dataset_path))["train"]
        ids_path = ids_path or str(BASE_DIR / "reproduction/artifacts/hotpotqa_seed42_n1000/ids.json")
        with open(ids_path, encoding="utf-8") as stream:
            self.frozen_ids = json.load(stream)
        if len(self.frozen_ids) != SAMPLE_SIZE or len(set(self.frozen_ids)) != SAMPLE_SIZE:
            raise ValueError(f"Expected exactly {SAMPLE_SIZE} unique frozen question IDs")
        source_ids = list(source_dataset["id"])
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("Duplicate source question IDs")
        source_index = {qid: index for index, qid in enumerate(source_ids)}
        missing = set(self.frozen_ids) - set(source_index)
        if missing:
            raise ValueError(f"Frozen IDs absent from prediction input: {sorted(missing)[:5]}")
        self.dataset = source_dataset.select([source_index[qid] for qid in self.frozen_ids])
        self.dbm = get_db_manager
        self.graph_helper = GraphHelper()
        self.llm = get_llm_model()
        self.emb = get_embeddings_model()
        self.top_k = top_k
        self.output_dir = output_dir
        self.sim_threshold = sim_threshold

        # Resume is by completed frozen ID in run_on_dataset, never by last row.

    def prepare(self):
        """
        读取 predictions.json 获取最后一个 question id，
        保存到 self.last_question_id，
        然后根据这个 id 截取 dataset，返回新的 dataset。
        """
        predictions_path = os.path.join(self.output_dir, "predictions.json")
        if not os.path.exists(predictions_path):
            print("predictions.json not found, using full dataset")
            return self.dataset

        with open(predictions_path, "r", encoding="utf-8") as f:
            predictions = json.load(f)

        # 取最后一个 question id
        if "answer" not in predictions or not predictions["answer"]:
            print("No answers found in predictions.json, using full dataset")
            return self.dataset

        # 取最后一个 key
        last_qid = list(predictions["answer"].keys())[-1]
        self.last_question_id = last_qid

        # 找到这个 id 在 dataset 中的 index
        id_list = list(self.dataset["id"])
        if last_qid in id_list:
            idx = id_list.index(last_qid)
            # 截取 idx 后的数据
            self.dataset = self.dataset.select(range(idx + 1, len(id_list)))
        else:
            print(f"last_question_id {last_qid} not found in dataset, using full dataset")

        return self.dataset

    # --------------------------
    # Utilities: LLM prompts
    # --------------------------
    def extract_entities_from_question(self, question: str, max_entities: int = 10) -> List[str]:
        """
        使用 LLM 提取问题中的所有关键实体（高召回）。
        """
        prompt = f"""
            Extract all key named entities from the question 
             ensuring that all extracted entities strictly come from the original text.

            Return a JSON array of strings (only the array). 
            Example: 
            When did the city where Greenwood Experimental School is located become the capital of the state where the screenwriter of Poor Fool was born?
            ["Greenwood Experimental School", "Poor Fool", "capital", "screenwriter"]
            
            Where was the place of death of Lambert I, Count Of Louvain's father?
            ["Lambert I, Count Of Louvain"]
            
            Extract entities only, do not extract relations.
            Question: {question}
            Entities:
            """

        response = self.llm.invoke(prompt).content

        # try to parse JSON in response
        entities = []
        try:
            start = response.find("[")
            if start != -1:
                end = response.find("]", start)
                if end != -1:
                    list_str = response[start:end + 1]
                    entities = json.loads(list_str)
                    if not isinstance(entities, list):
                        entities = []
        except Exception:
            # fallback: simple splitting by commas on capitalized words or quoted terms
            words = question.replace("?", "").replace(".", "").split()
            candidates = [w for w in words if w[0].isupper() or "'" in w]
            entities = candidates[:max_entities]
        # dedupe & trim
        entities = list(dict.fromkeys([e.strip().strip('"').strip("'") for e in entities if e and len(e.strip()) > 0]))[
                   :max_entities]
        return entities

    def decompose_question(self, qid: str, ori_question: str, question_type: str, entities: List[str], context_chunks: List) -> List[
        SubQuestion]:
        """
        给定初始上下文，使用 LLM 对问题进行分解成有序子问题。
        返回 SubQuestion 列表（按执行顺序）。
        """
        context = "\n".join([context.get("text", "") for context in context_chunks])

        ent_list_str = ", ".join([f'"{e}"' for e in entities])
        prompt = f"""
        Decompose a complex question into a minimal, ordered sequence of smaller, sequentially executable one-hop sub-questions that together answer the original question.
        Each sub-question must:
        - Use only the provided entities.
        - Specify the entities used.
        - Keep the number of sub-questions as small as possible while still allowing a deterministic reconstruction of the final answer.
        - Indicate whether it requires results from previous sub-questions (needs_external: true/false).
        - Be a one-hop simple question, answerable with a single direct query or fact lookup.
        - Be clear, concise, and directly contribute to answering the main question.
        
        Context
        This context helps you understand the general background and scenario of the question.
        {context}
        
        Output:
        - A JSON array of sub-questions, where each sub-question is an object with:
          - question: The sub-question text (ending with a question mark).
          - entities: List of entities used in the sub-question.
          - needs_external: Boolean indicating if the sub-question depends on results from prior sub-questions.

        Example:
        For the question: "When did the city where Greenwood Experimental School is located become the capital of the state where the screenwriter of 'Poor Little Fool' was born?"
        Entities: ["Greenwood Experimental School", "Poor Little Fool", "capital"]
        Output:
        [
          {{"question": "In which city is Greenwood Experimental School located?", "entities": ["Greenwood Experimental School"], "needs_external": false}},
          {{"question": "Who is the screenwriter of 'Poor Little Fool'?", "entities": ["Poor Little Fool"], "needs_external": false}},
          {{"question": "In which state was the screenwriter of 'Poor Little Fool' born?", "entities": ["Poor Little Fool"], "needs_external": true}},
          {{"question": "When did that city become the capital of that state?", "entities": ["capital"], "needs_external": true}}
        ]
        
        Are Random House Tower and 888 7th Avenue both used for real estate?
        What is Random House Tower used for?
        What is 888 7th Avenue used for?
        
        Are Giuseppe Verdi and Ambroise Thomas both Opera composers
        What type of composer was Giuseppe Verdi?
        What type of composer was Ambroise Thomas?
        
        Which Australian city founded in 1838 contains a boarding school opened by a Prime Minister of Australia and named after a school in London of the same name.
        Which Australian city was founded in 1838?
        Which boarding school in that city was opened by a Prime Minister of Australia and named after a school in London of the same name?
        
        Rostker v. Goldberg held that the practice of what way of filling armed forces vacancies was consitutional?
        What way of filling armed forces vacancies was addressed in Rostker v. Goldberg?
        Which way of filling armed forces vacancies was held constitutional in Rostker v. Goldberg?
        
        Who is the maternal grandfather of Antiochus X Eusebes?
        Who was the mother of Antiochus X Eusebes?
        Who was the father of [mother’s full name]?
        
        'Father-in-law’ can be decomposed into 'spouse' and 'father'

        Input:
        - Entities: [{ent_list_str}]
        - Question: {ori_question}
        """
        raw = self.llm.invoke(prompt).content

        # parse JSON
        subs: List[SubQuestion] = []
        try:
            start = raw.find('[')
            end = raw.rfind(']') + 1
            arr = raw[start:end]
            parsed = json.loads(arr)
            for i, item in enumerate(parsed):
                if isinstance(item, dict):
                    question = item.get("question", "")
                    ents = item.get("entities", [])
                    needs = bool(item.get("needs_external", False))
                    subs.append(SubQuestion(id=i, ori_question_id=qid, ori_question=ori_question, question=question,
                                            question_type=question_type,
                                            entities=ents,
                                            needs_external=needs))
                else:
                    subs.append(SubQuestion(id=i, ori_question_id=qid, ori_question=ori_question, question=str(item),
                                            question_type=question_type, entities=[],
                                            needs_external=False))
        except Exception as e:
            # fallback: trivial single step
            subs.append(SubQuestion(id=0, ori_question_id=qid, ori_question=ori_question, question=ori_question,
                                    entities=entities, question_type=question_type,
                                    needs_external=False))
            logger.warning("Decompose parsing failed, fallback to single-step. Error: %s", e)
        return subs

    # --------------------------
    # KG lookup & retrieval
    # --------------------------
    def entity_similarity_linking(self, entity_names: List[str], question_id: str) -> List[Dict[str, Any]]:
        """
        对每个实体名：直接用向量检索KG中相似实体（基于 {ENTITY_LABEL}），阈值为0.9。
        使用question_id来过滤
        """
        resolved = []
        for name in entity_names:
            try:
                name_emb = np.array(self.emb.embed_query(name), dtype=float)
                # 直接用向量搜索实体节点
                candidates = self.graph_helper.get_nodes_with_embeddings(name_emb, label=ENTITY_LABEL)
                # 过滤阈值，取最多一个
                if candidates:
                    filtered = [c for c in candidates if
                                c.get("entity_id").split("__" ,1)[1] == question_id]  # 该question_id的实体
                    if filtered:
                        c = filtered[0]
                        resolved.append({
                            "query_name": name,
                            "entity_id": c["entity_id"],
                            "description": c["description"],
                            "entity_type": c["entity_type"],
                            "title": c["title"],
                            "score": c["score"]
                        })
                else:
                    resolved.append({
                        "query_name": name,
                        "entity_id": None,
                        "description": name,
                        "score": None
                    })
            except Exception as e:
                logger.debug("Entity resolution for '%s' failed: %s", name, e)
                resolved.append({
                    "query_name": name,
                    "entity_id": None,
                    "description": name,
                    "score": None
                })
        return resolved

    def retrieve_chunks_for_subq(self, subq_text, question_id: str, top_k: int = None) -> \
            List[Dict[str, Any]]:
        """
        将问题向量化，去 __Chunk__ 节点中检索相似的chunk。
        修改：适配新的chunk属性结构
        """
        top_k = top_k or self.top_k
        try:
            q_emb = np.array(self.emb.embed_query(subq_text), dtype=float)
            all_chunks = self.graph_helper.get_nodes_with_embeddings(q_emb, label=CHUNK_LABEL, qid=question_id)
            # all_chunks = [c for c in all_chunks if c.get("fileName").split("_")[1] == question_id]  # 该question_id的实体

            # 3️⃣ 切分 chunk 为段落，并保留 title
            texts, chunk_refs, titles = [], [], []
            for c in all_chunks:
                title = c.get("title", "")
                for line in c.get("text", "").split("\n"):
                    line = line.strip()
                    texts.append(f"{title}\n{line}")  # ✅ 在文本前加 title
                    chunk_refs.append(c)
                    titles.append(title)

            # 4️⃣ 批量向量化所有段落
            para_embs = np.array(self.emb.embed_documents(texts), dtype=float)

            # 5️⃣ 计算相似度
            q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-8)
            para_norm = para_embs / (np.linalg.norm(para_embs, axis=1, keepdims=True) + 1e-8)
            sims = np.dot(para_norm, q_norm)

            # 6️⃣ 选出 top_k chunk（按段落相似度排序）
            top_idx = np.argsort(sims)[::-1]
            seen, filtered_chunks = set(), []
            for i in top_idx:
                c = chunk_refs[i]
                if c["id"] not in seen:
                    seen.add(c["id"])
                    filtered_chunks.append(c)
                if len(filtered_chunks) >= top_k:
                    break

            return filtered_chunks

        except Exception as e:
            print("Chunk retrieval failed:", e)
            return []

    def retrieve_entity_subgraph(self, entity_id: str):
        """
        获取某个实体的一跳路径（仅包含与其他实体的连接）。
        返回格式: [(head_id, relation, tail_id), ...]
        """
        onehop = self.graph_helper.get_entity_one_hop(entity_id)
        if not onehop:
            return []

        entity_id = onehop.get("entity_id", "")
        neighbors = onehop.get("neighbors", [])
        paths = []

        for n in neighbors:
            rel = n.get("rel", "")
            neighbor_data = n.get("neighbor", {})
            neighbor_label = neighbor_data.get("label", "")

            # 过滤掉 Document 和 Chunk
            if neighbor_label in {"__Document__", "__Chunk__"}:
                continue

            if neighbor_label == "__Entity__":
                entity_id = entity_id.split("__")[0]
                neighbor_id = neighbor_data.get("entity_id", "").split("__")[0]
                paths.append(f"{entity_id} ({onehop.get('description')}) --{rel}-- {neighbor_id} ({neighbor_data.get('description')})")

        return paths

    # --------------------------
    # QA step: ask LLM with context
    # --------------------------
    def rewrite_question_with_context(self, subq: SubQuestion, pre_qa) -> SubQuestion:
        """
        使用前置答案重新整理需要外部信息的问题，使其变成不需要前置信息的问题,
        TODO 可以将入参统一为List[SubQuestion]
        """

        if not subq.needs_external or not pre_qa:
            return subq

        pre_questions_info = [{"question": sub.question, "entities": sub.entities, "needs_external": sub.needs_external,
                               "answer": sub.answer} for sub in pre_qa]
        cur_questions_info = {"current_subquestion": subq.question, "entities": subq.entities,
                              "needs_external": subq.needs_external}

        prompt = f"""
        You are given:
        1) the original complex question (original_question)
        2) a list of decomposed sub-questions (sub_questions). Each sub-question is a JSON object with keys:
           - sub-question (string)
           - entities (list of strings)
           - needs_external (boolean)  # false if this sub-question originally required answers from prior steps
           - answer (string or null)
        3) the current sub-question to rewrite

        ## Task:
        Using the original question and the full list of sub-questions (with their answers and metadata), rewrite the specified current sub-question so that it is:
        - self-contained (does not require other sub-question answers),
        - single-hop (can be answered by reading a single factual sentence/record),
        - faithful to the original intent (do not change the logical meaning),
        - not introducing any facts that are NOT present in the provided answers.

        ## rules:
        - If the current sub-question refers to an entity that was resolved in a previous sub-question (e.g., "the author of X"), replace that reference with the actual resolved entity name from earlier answers.
        - Update the entities list so it reflects the concrete entity names referenced in the rewritten question.
        - After rewriting, the returned needs_external must be false if the rewrite is fully self-contained; if the previous answers do NOT provide sufficient information to make it self-contained, return the best-effort rewrite and set needs_external to true.
        - Do NOT invent or hallucinate new facts. Use only information available in the given sub-question answers.
        - Output **JSON only**, no explanations, no extra text.

        ## Example:
        original_question: "When did the city where Greenwood Experimental School is located become the capital of the state where the author of 'The Great Gatsby' was born?"

        sub_questions:
        [{{"question": "In which city is Greenwood Experimental School located?", "entities": ["Greenwood Experimental School"], "needs_external": false, "answer": "Springfield"}},
        {{"question": "Who is the author of 'The Great Gatsby'?", "entities": ["The Great Gatsby"], "needs_external": false, "answer": "F. Scott Fitzgerald"}}]

        {{"current_subquestion": "In which state was the author of 'The Great Gatsby' born?", "entities": ["The Great Gatsby"], "needs_external": true, "answer": null}}

         Example expected output for the above input (JSON only):
        {{"rewritten_text": "In which state was F. Scott Fitzgerald born?", "updated_entities": ["F. Scott Fitzgerald"], "needs_external": false}}

        Strict output JSON schema :
        {{
        "rewritten_text": "<self-contained single-hop question text>",
        "updated_entities": ["entity_name_1", "entity_name_2"],
        "needs_external": <true|false>
        }}

        Input:
        original_question: {subq.ori_question}

        sub_questions:
        {pre_questions_info}

        {cur_questions_info}
        Output:
        """

        try:

            raw = self.llm.invoke(prompt).content
            start = raw.find('{')
            end = raw.rfind('}') + 1
            parsed = json.loads(raw[start:end])

            # 创建新的重写后的子问题
            rewritten_subq = SubQuestion(
                id=subq.id,
                ori_question_id=subq.ori_question_id,
                ori_question=subq.ori_question,
                question=parsed.get("rewritten_text", subq.question),
                question_type=subq.question_type,
                entities=parsed.get("updated_entities", subq.entities),
                needs_external=False,  # 重写后不再需要外部信息
                solved=False
            )
            return rewritten_subq

        except Exception as e:
            logger.warning(f"Failed to rewrite subquestion {subq.id}: {e}")
            # 回退：直接将前置答案拼接到问题中
            combined_text = f"Given that {pre_qa[-1].question, pre_qa[-1].answer} , {subq.question}"
            rewritten_subq = SubQuestion(
                id=subq.id,
                ori_question_id=subq.ori_question_id,
                ori_question=subq.ori_question,
                question=combined_text,
                question_type=subq.question_type,
                entities=subq.entities,
                needs_external=False,
                solved=False
            )
            return rewritten_subq

    def answer_subquestion(self, subq: SubQuestion, resolved_entities: List[Dict[str, Any]]) -> SubQuestion:
        """
        对单个子问题做检索+LLM回答。
        修改：移除prev_answers参数，因为问题已经被重写为自包含的
        """
        # Collect relevant chunks from entities and question similarity
        all_chunks = []
        subgraph_text = []
        for ent_name in subq.entities:
            match = next((r for r in resolved_entities if r["query_name"] == ent_name), None)
            if match and match.get("entity_id"):
                eid = match["entity_id"]
                subgraph_text = self.retrieve_entity_subgraph(eid)
                related_chunks = self.graph_helper.get_entity_chunks(eid)

                all_chunks.extend(related_chunks)

        if all_chunks:
            q_emb = np.array(self.emb.embed_query(subq.question), dtype=float)
            all_chunks = self.graph_helper.find_most_similar_chunk([chunks["id"] for chunks in all_chunks], q_emb)
        # if all_chunks:
        #     print()
        #     print("eid--------------------------------related_chunks:", all_chunks[0]["id"], len(all_chunks))
        #     print()
        # Add question similarity chunks
        q_chunks = self.retrieve_chunks_for_subq(subq.question, subq.ori_question_id, self.top_k)
        all_chunks.extend(q_chunks)

        # 去重
        seen_chunks = {}
        for chunk in all_chunks:
            chunk_key = chunk['title']
            if chunk_key not in seen_chunks:
                seen_chunks[chunk_key] = chunk
        # print()
        # print("-------------------------去重：",subq.ori_question_id, len(all_chunks)-len(q_chunks), len(q_chunks), len(list(seen_chunks.keys())))
        # print()

        all_chunks = list(seen_chunks.values())
        context_parts = []
        # 整和成格式良好的片段 title index
        for chunk in all_chunks:
            sentences = chunk.get('text', '').split("\n")
            context = "title : " + chunk['title'] + "\n"
            for i, sentence in enumerate(sentences):
                context = context + f"index : {i}  " + sentence
                context += "\n"
            context_parts.append(context)
        context = "\n\n".join(context_parts)

        subgraph_text = "\n".join(subgraph_text)
        prompt = f"""Answer the sub-question based on the context. Provide all possible answers and list supporting chunk titles if possible.
                Sub-question: {subq.question}
                - If there is conflict ,choose **most likely** answer
                - The answer must come from nouns or phrases in the original text.
                - Briefly explain your reasoning in a few sentences before providing the JSON.
                
                Context:
                {context}
                
                Return : 
                reason:...
                {{"answer":"...","supporting_title":["title1", "title2"]}}
                If there are no supporting_titles then return {{"answer":"...","supporting_title":[]}}
                """

        raw = self.llm.invoke(prompt).content

        # Parse
        try:
            start = raw.find('{')
            end = raw.rfind('}') + 1
            parsed = json.loads(raw[start:end])
            subq.answer = parsed.get("answer", "UNANSWERABLE")

            supporting_chunks = []
            for supporting_title in parsed.get("supporting_title", []):
                chunk = seen_chunks.get(supporting_title)
                if chunk is not None:
                    supporting_chunks.append(chunk)

            subq.supporting_chunks = supporting_chunks
            subq.solved = True  # 标记为已解决
        except Exception:
            subq.answer = raw.strip()
            subq.solved = True  # 即使解析失败也标记为已解决，避免无限循环

        return subq

    def answer_question_2hop(self,qid, question,entities: List[str], resolved_entities):
        """尝试根据2条路径来回答问题"""
        final_answer = ""
        supporting_facts = []
        paths = []
        graph_context = set()
        global len_hoppath
        for entity in entities:
            match = next((r for r in resolved_entities if r["query_name"] == entity), None)
            if match and match.get("entity_id"):
                eid = match["entity_id"]
                paths.extend(self.graph_helper.get_2hop_path(eid))
                len_hoppath[0] += len(paths)
                len_hoppath[1] += 1
        for path in paths[:10]:
            e1, e1_desc = path.get("e1", "").split("__")[0], path.get("e1_desc", "")
            rel1, rel1_des = path.get("rel1", "").lower(), path.get("rel1_desc", "")
            e2, e2_desc = path.get("e2", "").split("__")[0], path.get("e2_desc", "")
            rel2, rel2_desc = path.get("rel2", "").lower(), path.get("rel2_desc", "")
            e3, e3_desc = path.get("e3", "").split("__")[0], path.get("e3_desc", "")
            graph_context.add(rel1_des)
            graph_context.add(rel2_desc)
        graph_context = "\n".join(graph_context)
        prompt = f"""给定指定实体的两跳路径信息，给出详细的推理过程判断是否能够回答问题，
        如果能够回答问题则返回json格式
        reason:...
        {{"answer":"...(entity name)"}}
        
        否则返回
        reason:...
        {{"answer":"None"}}
        
        Context:
        {graph_context}
        
        Question:
        {question}
        """

        raw_final = self.llm.invoke(prompt).content

        try:
            start = raw_final.find('{')
            end = raw_final.rfind('}') + 1
            parsed = json.loads(raw_final[start:end])
            final_answer = parsed.get("answer", final_answer)
        except Exception:
            final_answer = "None"
        return final_answer, supporting_facts
    # --------------------------
    # High-level QA for one question
    # --------------------------
    def qa_for_one(self, sample: Dict[str, Any]) -> QAResult:
        """
        使用while循环处理子问题，支持动态重写需要前置信息的问题
        """
        qid = sample.get("_id") or sample.get("id") or str(time.time())
        question = sample.get("question") or sample.get("query") or ""
        question_type = sample.get("type")
        logger.info("Processing QID=%s: %s", qid, question[:80])

        # 1) extract key entities
        entity_names = self.extract_entities_from_question(question)
        resolved = self.entity_similarity_linking(entity_names, qid)

        # 1.1)根据2跳路径尝试提前回答
        final_answer, supporting_facts = self.answer_question_2hop(qid, question, entity_names, resolved)
        # final_answer="None"
        if final_answer=="None":
            # 2) 得到语境上下文
            context_chunks = self.retrieve_chunks_for_subq(question, qid, 2)

            # 3) decompose question
            subs = self.decompose_question(qid, question, question_type, entity_names, context_chunks)

            # 4) 使用while循环处理所有子问题
            pre_qa = []
            max_iterations = len(subs) * 2  # 防止无限循环
            iteration = 0


            while any(not sub.solved for sub in subs) and iteration < max_iterations:
                iteration += 1
                progress_made = False

                # 从列表头开始遍历
                for i, sub in enumerate(subs):
                    if sub.solved:
                        continue
                    pre_qa = subs[:i]
                    # 如果不需要外部信息，直接处理
                    if not sub.needs_external:
                        logger.info(f"Processing sub-question {sub.id} (independent): {sub.question}")
                        subs[i] = self.answer_subquestion(sub, resolved)
                        progress_made = True
                        break  # 处理一个后重新开始遍历

                    # 如果需要外部信息，检查是否有足够的前置答案
                    else:
                        # 检查是否有可用的前置答案
                        if pre_qa:
                            logger.info(f"Processing sub-question {sub.id} (with context): {sub.question}")
                            # 重写问题，使其包含前置信息
                            rewritten_sub = self.rewrite_question_with_context(sub, pre_qa)
                            logger.info(f"Processing rewrite sub-question {sub.id} (independent): {rewritten_sub.question}")
                            # 链接新实体并加入解析列表
                            resolved.extend(self.entity_similarity_linking(rewritten_sub.entities, qid))
                            # 处理重写后的问题
                            processed_sub = self.answer_subquestion(rewritten_sub, resolved)
                            # 保持原始ID和solved状态
                            processed_sub.id = sub.id
                            subs[i] = processed_sub
                            # prev_answers[sub.id] = processed_sub.answer or "UNANSWERABLE"
                            progress_made = True
                            break  # 处理一个后重新开始遍历

                # 如果一轮遍历没有任何进展，避免无限循环
                if not progress_made:
                    logger.warning("No progress made in this iteration, forcing completion of remaining questions")
                    for i, sub in enumerate(subs):
                        if not sub.solved:
                            # 强制处理剩余问题，即使它们需要外部信息
                            rewritten_sub = self.rewrite_question_with_context(sub, pre_qa)
                            processed_sub = self.answer_subquestion(rewritten_sub, resolved)
                            processed_sub.id = sub.id
                            subs[i] = processed_sub
                            # prev_answers[sub.id] = processed_sub.answer or "UNANSWERABLE"
                    break

            # 确保所有问题都已解决
            for sub in subs:
                if not sub.solved:
                    sub.solved = True
                    sub.answer = "UNANSWERABLE"

            # 4) Synthesize final answer using all sub-answers
            context_parts = []
            for sub in subs:
                ans = sub.answer or "UNANSWERABLE"
                context_parts.append(f"SubQ : {sub.question} SubAns[{sub.id}]: {ans}")

                # 整和成格式良好的片段 title index
                for chunk in sub.supporting_chunks:
                    sentences = chunk.get('text', '').split("\n")
                    context = "title : " + chunk.get('title', "") + "\n"
                    for i, sentence in enumerate(sentences):
                        context = context + f"index : {i}  " + sentence
                        context += "\n"
                    context_parts.append(context)

            if len(context_parts) == 1:
                for chunk in context_chunks:
                    sentences = chunk.get('text', '').split("\n")
                    context = "title : " + chunk.get('title', "") + "\n"
                    for i, sentence in enumerate(sentences):
                        context = context + f"index : {i}  " + sentence
                        context += "\n"
                    context_parts.append(context)

            sub_answers_text = "\n\n".join(context_parts)

            final_prompt = f"""
    
            In the original HotpotQA paper, the authors distinguish between a few reasoning types:
    
            ### Bridge-type
    
            - **Typical Answer**: A specific entity (e.g., person, location, work).
            - Example:
              "Where was the author of *Arrowsmith* born?"
    
              Step 1: Find the author of *Arrowsmith* → Sinclair Lewis
              Step 2: Find Sinclair Lewis’s birthplace → Minnesota
    
            ---
    
            ### Comparison-type
    
            - **Typical Answer**: Two sub-categories exist:
    
              1. **Entity selection** – choose the correct one between two candidates.
    
                 - Example:
                   "Which magazine was started first, Arthur’s Magazine or First for Women?"
    
                   Answer: Arthur’s Magazine
              2. **yes/no judgment** – usually phrased as questions starting with *Do*, *Does*, or *Are*.
    
                 - Example:
                   "Are the Nile and Amazon rivers located on the same continent?"
    
                   Answer: no
    
            You will be given:
            - Original Question (string)
            - question type
            - Sub-question Answers (array of strings)
            - Candidate Chunks (optional, array with title and implicit 0-based index)
    
            ## 1. Reasoning paragraph
            Write a short reasoning paragraph (max 3 sentences).  
            - Identify keywords, entities, relations in the original question.  
            - - Map sub-question answers to these entities/relations, together with the title context, relying not only on sub-question answers but also on the full text.
            - If there is conflict, briefly explain why you choose one conclusion (or say "most likely").  
            - For example, the keyword **government position** should be understood as referring to a clearly defined official role within the internal structure of a government institution.
    
            ## 2. JSON output
            Immediately after the reasoning paragraph, output a single valid JSON object, nothing else.  
            Schema:
    
            {{"answer":"...(Give only a concise answer, extracted as a short word, phrase, or sentence directly from the original text.)","supporting_chunks":[["title1", index(int)],["title2", index(int)]]}}
            If there are no supporting_chunks, then return {{"answer":"...(Give only a concise answer, extracted as a short word, phrase, or sentence directly from the original text.)","supporting_chunks":[]}}
            **If the answer is supported by any reference facts, then add the corresponding title to supporting_chunks."**
    
            Rules:
            - "supporting_chunks" is an array of [title, index], where index is 0-based.  
            - If no supporting chunks, return an empty array.  
            - Keep all numbers, units, qualifiers, and modifiers as they appear.  
            - The answer must be copied verbatim from the passage, or as a concatenation of verbatim spans.  
            - Do not normalize text (e.g., "Oct" → "October", "3,677" → "3677").  
            - If there is conflict ,choose **most likely** answer
            - Give only a concise answer, extracted as a short word, phrase, or sentence directly from the original text.
            ---
    
            ### Example 1
            Context: "He served from 1981 to 2011."
            Question: "When did he serve?"
            ❌ Bad answer: "1981 to 2011"
            ✅ Good answer: "from 1981 to 2011"
    
            ### Example 2
            Context: "The stadium opened in October 1921."
            Question: "When did the stadium open?"
            ❌ Bad answer: "1921"
            ✅ Good answer: "October 1921"
            
            ### Example 3
            Question: "Why did Grand Duke Kirill Vladimirovich Of Russia's wife die?"
            ❌ Bad answer: "Victoria died after suffering a stroke"
            ✅ Good answer: "stroke"
    
            ### Example 4
            Question: "Where was the place of death of Lambert I, Count Of Louvain's father?"
            ❌ Bad answer: "in exile in Bohemia"
            ✅ Good answer: "Bohemia"
            
            ### Example 5
            Question: "Where did Prince Gustav Of Thurn And Taxis (1848–1914)'s mother die?"
            ❌ Bad answer: "Obermais, Meran, County of Tyrol, Austria–Hungary"
            ✅ Good answer: "Meran"
            
            Input:
            Original question: {question}
            question type: {question_type}
            Sub-question answers:
            Give only a concise answer, extracted as a short word, phrase, or sentence directly from the original text.
            {sub_answers_text}
            
            Original question: {question}
            Answer:
                        """

            raw_final = self.llm.invoke(final_prompt).content

            supporting_facts = []
            final_answer = "UNANSWERABLE"
            try:
                start = raw_final.find('{')
                end = raw_final.rfind('}') + 1
                parsed = json.loads(raw_final[start:end])
                final_answer = parsed.get("answer", final_answer)
                supporting_facts = parsed.get("supporting_chunks", [])

            except Exception:
                final_answer = raw_final.strip()

            debug = {
                "extracted_entities": resolved,
                "sub_questions": [s.__dict__ for s in subs],
                "final_raw": raw_final,
                "iterations": iteration
            }
        else:
            print(f"--------{qid}提前回答！！！--------")
            global pre_id
            pre_id.append(qid)
        return QAResult(qid=qid, answer=final_answer, sp=supporting_facts)

        # --------------------------
        # Batch pipeline entry
        # --------------------------

    def _process_sample(self, sample):
        """
        处理单个样本的逻辑（包含重试机制）。
        这个方法将被并行调用。
        """
        qid = sample.get("id", "unknown")
        started = time.perf_counter()
        errors = []
        for attempt in range(3):
            if (run_root() / "STOP_REQUESTED").exists() or (run_root() / "FATAL_PROVIDER_ERROR.json").exists():
                break
            try:
                with call_context(question_id=qid, question_attempt=attempt + 1):
                    result = self.qa_for_one(sample)
                if result.qid != qid:
                    raise ValueError("QA returned a different question ID")
                return {
                    "id": qid, "answer": result.answer,
                    "supporting_facts": result.sp, "type": sample.get("type", "unknown"),
                    "status": "completed", "attempts": attempt + 1,
                    "elapsed_seconds": time.perf_counter() - started,
                    "prior_errors": errors, "early_two_hop_answer": qid in pre_id,
                }
            except Exception as error:
                errors.append({"attempt": attempt + 1, "type": type(error).__name__, "message": str(error)})
                logger.error("QID=%s attempt=%s failed: %s", qid, attempt + 1, error)
        return {
            "id": qid, "answer": "", "supporting_facts": [],
            "type": sample.get("type", "unknown"), "status": "failed",
            "attempts": len(errors), "elapsed_seconds": time.perf_counter() - started,
            "errors": errors,
        }


    def run_on_dataset(self):
        """Run scoped text workers, checkpoint each ID, preserve frozen order."""
        os.makedirs(self.output_dir, exist_ok=True)
        p1 = os.path.join(self.output_dir, "predictions.json")
        p2 = os.path.join(self.output_dir, "predictions_with_supporting_facts.json")
        by_id = {}
        if os.path.exists(p2):
            with open(p2, encoding="utf-8") as stream:
                previous = json.load(stream)
            for result in previous:
                if result["id"] in by_id or result["id"] not in self.frozen_ids:
                    raise ValueError("Duplicate or unexpected checkpoint question ID")
                by_id[result["id"]] = result

        def checkpoint():
            ordered = [by_id[qid] for qid in self.frozen_ids if qid in by_id]
            write_json(p2, ordered)
            write_json(p1, {
                "answer": {row["id"]: row["answer"] for row in ordered},
                "sp": {row["id"]: row["supporting_facts"] for row in ordered},
            })
            write_json(os.path.join(self.output_dir, "progress.json"), {
                "total": SAMPLE_SIZE, "recorded": len(ordered),
                "completed": sum(row["status"] == "completed" for row in ordered),
                "failed": sum(row["status"] != "completed" for row in ordered),
            })

        pending = [
            sample for sample in self.dataset
            if by_id.get(sample["id"], {}).get("status") != "completed"
        ]
        with ThreadPoolExecutor(max_workers=TEXT_DISPATCH_WORKERS) as executor:
            futures = {executor.submit(self._process_sample, sample): sample["id"] for sample in pending}
            for future in tqdm(as_completed(futures), total=len(futures), desc="hotpotqa KG QA"):
                result = future.result()
                by_id[result["id"]] = result
                append_jsonl(os.path.join(self.output_dir, "question_events.jsonl"), result)
                checkpoint()
        checkpoint()
        if set(by_id) != set(self.frozen_ids):
            raise RuntimeError("Prediction coverage differs from the frozen 1000 IDs")
        return [by_id[qid] for qid in self.frozen_ids]


# ------------------------------
# If run as script
# ------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hotpot QA over Knowledge Graph pipeline")
    parser.add_argument("--dataset_path", type=str,
                        default="../dataset/2wiki/validation-00000-of-00001.parquet",
                        help="Path to parquet file (hotpot distractor validation)")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--ids_path", type=str, default=None, help="Frozen question ID file")
    args = parser.parse_args()

    # initialize pipeline
    pipeline = HotpotKGPipeline(dataset_path=args.dataset_path, output_dir=args.output_dir, ids_path=args.ids_path)
    pipeline.run_on_dataset()
    if len_hoppath[1] != 0:
        print("平均每个实体的2跳实体个数为：", len_hoppath[0]/len_hoppath[1], len_hoppath[1])
    print(pre_id)

