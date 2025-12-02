import ast
import os
import re
import time
from itertools import combinations
from typing import List, Dict, Any, Set

import numpy as np
import psutil
import shutup
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.text import Text

from config.neo4jdb import get_db_manager
from config.prompt import (
    system_template_build_graph,
    human_template_build_graph
)
from config.settings import (
    entity_types,
    relationship_types,
    theme,
    DATASET_DIR,
    CHUNK_SIZE,
    OVERLAP,
    MAX_WORKERS, BATCH_SIZE,
)
from graph import EntityRelationExtractor
from graph import GraphStructureBuilder
from graph import GraphWriter
from model.get_models import get_llm_model, get_embeddings_model
from processor.dataset_processor import DatasetProcessor

shutup.please()


class KnowledgeGraphBuilder:
    """
    知识图谱构建器，负责图谱的基础构建流程。
    
    主要功能包括：
    1. 文件读取和解析
    2. 文本分块
    3. 实体和关系抽取
    4. 构建基础图结构
    5. 写入数据库
    """

    def __init__(self):
        """初始化知识图谱构建器"""
        # 初始化终端界面
        self.console = Console()
        self.processed_documents = []

        # 添加计时器
        self.start_time = None
        self.end_time = None

        # 阶段性能统计
        self.performance_stats = {
            "初始化": 0.0,
            "文件处理": 0.0,  # 改为"文件处理"，包含读取和分块
            "图结构构建": 0.0,
            "实体抽取": 0.0,
            "写入数据库": 0.0
        }

        # 初始化组件
        self._initialize_components()

    def _create_progress(self):
        """创建进度显示器"""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self.console
        )

    def _initialize_components(self):
        """初始化所有必要的组件"""
        init_start = time.time()

        with self._create_progress() as progress:
            task = progress.add_task("[cyan]初始化组件...", total=4)

            # 初始化模型
            self.llm = get_llm_model()
            self.embeddings = get_embeddings_model()
            progress.advance(task)  # 进度条再前进 1/4

            # 初始化图数据库连接
            db_manager = get_db_manager()
            self.graph = db_manager.graph
            progress.advance(task)

            # 初始化文档处理器
            # self.document_processor = DocumentProcessor(FILES_DIR, CHUNK_SIZE, OVERLAP)
            # 初始化数据集处理器
            self.dataset_processor = DatasetProcessor(DATASET_DIR, CHUNK_SIZE, OVERLAP)
            progress.advance(task)

            self.struct_builder = GraphStructureBuilder(batch_size=BATCH_SIZE)
            self.entity_extractor = EntityRelationExtractor(
                self.llm,
                system_template_build_graph,
                human_template_build_graph,
                entity_types,
                relationship_types,
                max_workers=MAX_WORKERS,
                batch_size=5  # LLM批处理大小保持小一些以确保质量
            )

            # 输出使用的参数
            self.console.print(f"[blue]并行处理线程数: {MAX_WORKERS}[/blue]")
            self.console.print(f"[blue]数据库批处理大小: {BATCH_SIZE}[/blue]")

            progress.advance(task)

        self.performance_stats["初始化"] = time.time() - init_start

    def _display_stage_header(self, title: str):
        """显示处理阶段的标题"""
        self.console.print(f"\n[bold cyan]{title}[/bold cyan]")

    def _display_results_table(self, title: str, data: Dict[str, Any]):
        """显示结果表格"""
        table = Table(title=title, show_header=True)
        table.add_column("指标", style="cyan")
        table.add_column("值", justify="right")

        for key, value in data.items():
            table.add_row(key, str(value))

        self.console.print(table)

    def _format_time(self, seconds: float) -> str:
        """格式化时间为小时:分钟:秒.毫秒"""
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}.{int((seconds % 1) * 1000):03d}"

    def _pre_parse_entity_data(self):
        """
        遍历所有文档，将LLM返回的实体关系字符串预解析为结构化的字典。
        这个结构化字典将包含节点和关系信息，为后续的剪枝做准备。
        """
        self._display_stage_header("预解析实体关系字符串")

        node_pattern = re.compile(r'\("entity" : "(.+?)" : "(.+?)" : "(.+?)"\)')
        rel_pattern = re.compile(r'\("relationship" : "(.+?)" : "(.+?)" : "(.+?)" : "(.+?)" : (.+?)\)')

        for doc in self.processed_documents:
            if "entity_data" not in doc or not isinstance(doc["entity_data"], list):
                continue

            # 'structured_data' 将是一个列表，每个元素对应一个chunk的解析结果
            doc['structured_data'] = []
            for i, raw_text in enumerate(doc["entity_data"]):
                # 每个chunk的节点和关系
                chunk_nodes = {}
                chunk_relationships = []
                try:
                    # 1. 解析所有节点定义，获取它们的类型和描述
                    for match in node_pattern.findall(raw_text):
                        node_id, node_type, description = match
                        node_id = node_id + "__" + doc.get("question_id")
                        chunk_nodes[node_id] = {"type": node_type, "description": description, "title": doc["title"][i],
                                                "question_id": doc.get("question_id", "")}

                    # 2. 解析所有关系
                    for match in rel_pattern.findall(raw_text):
                        source_id, rel_type, target_id, description, weight = match
                        source_id = source_id + "__" + doc.get("question_id")
                        target_id = target_id + "__" + doc.get("question_id")

                        chunk_relationships.append({
                            "head": source_id,
                            "relation": rel_type,
                            "tail": target_id,
                            "description": description,
                            "weight": float(weight)
                        })
                        # 确保关系中涉及的实体也被添加到节点列表中（以防万一它们没有被'entity'标签定义）
                        if source_id not in chunk_nodes:
                            chunk_nodes[source_id] = {"type": "未知", "description": "从关系中推断",
                                                      "title": doc["title"][i],
                                                      "question_id": doc.get("question_id", "")}
                        if target_id not in chunk_nodes:
                            chunk_nodes[target_id] = {"type": "未知", "description": "从关系中推断",
                                                      "title": doc["title"][i],
                                                      "question_id": doc.get("question_id", "")}
                except Exception as e:
                    print("解析数据match数据：", match)
                    print(f"[解析错误] {e}")

                doc['structured_data'].append({
                    "nodes": chunk_nodes,
                    "relationships": chunk_relationships
                })

    def _merge_local_entities(self, similarity_threshold: float = 0.95):
        """
        对提取出的实体进行向量化，以减少稀疏性。
        此函数会遍历每个文档，并在文档内部完成实体合并。

        Args:
            similarity_threshold (float): 用于筛选候选合并对的相似度阈值。
        """
        self.console.print("\n[bold]对每个文档分别进行实体合并...[/bold]")

        for doc in self.processed_documents:
            self.console.print(f"\n[cyan]正在处理文件: {doc['filename']}...[/cyan]")

            structured_data_per_chunk = doc.get("structured_data", [])
            if not structured_data_per_chunk:
                self.console.print("[yellow]未发现结构化数据，跳过此文件。[/yellow]")
                continue

            # 1. 收集实体并计算其在文档中的出现频率
            local_entities_map = {}  # 用于计算频率
            entity_details_map = {}  # 用于查找实体的描述（如label, des）

            for chunk_data in structured_data_per_chunk:
                nodes = chunk_data.get("nodes", {})
                for entity_id, entity_data in nodes.items():
                    local_entities_map[entity_id] = local_entities_map.get(entity_id, 0) + 1
                    # 存储实体的详细数据，如果已存在则不覆盖
                    if entity_id not in entity_details_map:
                        entity_details_map[entity_id] = entity_data

            # 将 dict_keys 转换为 list 以便进行向量化
            local_entities = list(local_entities_map.keys())

            if len(local_entities) < 2:
                self.console.print("[green]实体数量不足，无需合并。[/green]")
                continue

            # 2. 向量化和候选对生成
            entity_embeddings = self.embeddings.embed_documents([entity.split("__")[0] for entity in local_entities])
            embedding_map = {entity: embedding for entity, embedding in zip(local_entities, entity_embeddings)}

            candidate_pairs = []
            for (entity1, emb1), (entity2, emb2) in combinations(embedding_map.items(), 2):
                cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                if cos_sim > similarity_threshold or (entity1 in entity2 or entity2 in entity1) :
                    candidate_pairs.append((entity1, entity2))

            if not candidate_pairs:
                self.console.print("[green]未发现相似实体对，无需合并。[/green]")
                continue
            self.console.print(f"发现 {len(candidate_pairs)} 对候选实体，提交给LLM确认...")
            self.console.print(f"候选实体对为：{candidate_pairs}")

            # 3. LLM 确认
            confirmed_merges = {}
            batch_size = 10
            for i in range(0, len(candidate_pairs), batch_size):
                batch = candidate_pairs[i:i + batch_size]

                # 2. 在生成提示时，为每个实体附加上下文描述
                # 使用一个更清晰的 for 循环来构建带描述的实体对列表
                formatted_pairs_lines = []
                for idx, (e1, e2) in enumerate(batch):
                    # 安全地获取实体描述 (label)
                    desc1 = (entity_details_map.get(e1, {}).get('label', '未知') + "\n" +
                             entity_details_map.get(e1,{}).get('description', ''))
                    desc2 = (entity_details_map.get(e2, {}).get('label', '未知') + "\n" +
                             entity_details_map.get(e2,{}).get('description', ''))
                    e1 = e1.split("__")[0]
                    e2 = e2.split("__")[0]
                    # 创建包含描述的行
                    line = f'{idx + 1}. 实体1: "{e1}" (详细信息: {str(desc1)}) | 实体2: "{e2}" (详细信息: {str(desc2)})'
                    formatted_pairs_lines.append(line)

                formatted_pairs = "\n".join(formatted_pairs_lines)

                # 3. [修改] 更新主提示，告知LLM新的格式并提供匹配的示例
                prompt = f"""你是一个实体识别和链接专家。你的任务是根据实体名称及其类型描述，判断每一对实体是否指向同一个真实世界的实体。

                请分析下面的实体对列表：
                {formatted_pairs}
    
                你的回答必须遵循以下规则：
                1.  返回一个单独的 Python 列表，其中包含布尔值（True 或 False）。
                2.  列表中的每个值必须与上面列表中的每个实体对一一对应。
                3.  除了这个 Python 列表，不要包含任何其他文字、解释或前缀。
    
                例如，对于输入：
                1. 实体1: "苹果公司" (详细信息: 公司) | 实体2: "Apple Inc." (详细信息: 公司)
                2. 实体1: "亚马逊" (详细信息: 公司) | 实体2: "亚马逊河" (详细信息: 地理位置)
    
                你的输出应该是：
                [True, False]
                """
                # Call the LLM
                response = self.llm.invoke(prompt).content

                # 2. [MODIFIED] Parse the Python list from the response.
                try:
                    list_str = None  # 初始化一个变量来存储找到的列表字符串

                    # 尝试1：使用更精确的非贪婪正则表达式
                    search_result = re.search(r"\[.*?\]", response)
                    if search_result:
                        list_str = search_result.group(0)

                    # 尝试2：如果正则失败，则回退到 find 方法
                    if not list_str:
                        start = response.find("[")
                        # 必须确保两个括号都找到了，并且顺序是正确的
                        if start != -1:
                            end = response.find("]", start)  # 从 start 位置之后开始找 ']'
                            if end != -1:
                                list_str = response[start:end + 1]  # 正确的切片

                    # 在两种方法都尝试过后，检查我们是否找到了列表字符串
                    if not list_str:
                        # 如果 list_str 仍然是 None，说明两种方法都失败了
                        raise ValueError("LLM 响应中未找到任何有效的 Python 列表 `[...]`。")

                    results_list = ast.literal_eval(list_str)

                    if not isinstance(results_list, list) or len(results_list) != len(batch):
                        raise ValueError("Parsed list is not valid or its length does not match the batch size.")

                    # 3. [MODIFIED] Iterate through the batch and the results list together.
                    for (e1, e2), is_match in zip(batch, results_list):
                        if is_match:
                            # This part of the logic remains the same
                            source, target = (e1, e2) if (local_entities_map.get(e1, 0) <
                                                          local_entities_map.get(e2, 0)) else (e2, e1)
                            final_target = confirmed_merges.get(target, target)
                            confirmed_merges[source] = final_target

                except (ValueError, SyntaxError) as e:
                    self.console.print(
                        f"[yellow]警告: 无法解析LLM对于批次 {i // batch_size + 1} 的响应。错误: {e}[/yellow]")
                    self.console.print(f"[yellow]LLM响应原文: {response}[/yellow]")
                    continue  # Skip to the next batch

            if not confirmed_merges:
                self.console.print("[green]LLM未确认任何合并操作。[/green]")
                continue
            self.console.print(f"LLM确认了 {len(confirmed_merges)} 个合并操作。正在更新图谱数据...")

            # 4. 更新节点(nodes)和关系(relationships)
            for chunk_data in structured_data_per_chunk:
                nodes = chunk_data.get("nodes", {})
                relationships = chunk_data.get("relationships", [])

                # 4.1 更新节点字典 (nodes)
                # 将被合并的节点(source)的数据迁移到目标节点(target)上，然后删除源节点
                nodes_to_delete = set()
                for source, target in confirmed_merges.items():
                    if source in nodes:
                        # 如果目标节点还不存在于这个chunk中，则把源节点的数据完整迁移过去
                        if target not in nodes:
                            nodes[target] = nodes[source]
                        # 如果目标节点已存在，可以选择性地合并属性,description
                        else:
                            nodes[target]["description"] += nodes[source].get("description")
                        # 标记源节点以便后续删除
                        nodes_to_delete.add(source)

                # 执行删除操作
                for node_id in nodes_to_delete:
                    if node_id in nodes:
                        del nodes[node_id]

                # 4.2 更新关系列表 (relationships)
                # 创建一个新的关系列表，只包含有效且非自循环的关系
                updated_relationships = []
                for rel in relationships:
                    # 获取关系中的头尾实体，并用合并字典找到它们的最终规范名称
                    head = confirmed_merges.get(rel.get("head"), rel.get("head"))
                    tail = confirmed_merges.get(rel.get("tail"), rel.get("tail"))

                    # 只有当头尾实体不同时，才保留这个关系
                    if head != tail:
                        rel["head"] = head
                        rel["tail"] = tail
                        updated_relationships.append(rel)

                # 用更新后的关系列表替换旧的列表
                chunk_data["relationships"] = updated_relationships

            self.console.print(f"[bold green]文件 '{doc['filename']}' 的实体合并完成！[/bold green]")

    def _build_entity_provenance_map(self) -> Dict[str, set]:
        """
        遍历所有处理过的文档，基于预解析的结构化数据，
        构建一个实体与其所在chunk_id集合的映射。
        """
        self._display_stage_header("构建实体出处映射 (Provenance Map)")

        provenance_map = {}

        for doc in self.processed_documents:
            chunks_with_ids = doc.get("graph_result", [])
            structured_data_per_chunk = doc.get("structured_data", [])

            if len(chunks_with_ids) != len(structured_data_per_chunk):
                continue  # 跳过不匹配的数据

            for i, chunk_data in enumerate(structured_data_per_chunk):
                chunk_id = chunks_with_ids[i].get("chunk_id")

                if not chunk_id:
                    continue
                # 遍历该chunk中出现的所有实体ID
                for entity_id in chunk_data.get("nodes", {}).keys():
                    provenance_map.setdefault(entity_id, set()).add(chunk_id)

        self.console.print(f"[green]实体-chunkid构建完成，共计 {len(provenance_map)} 个实体。[/green]")
        return provenance_map

    def _prune_triples_in_memory(self, provenance_map: Dict[str, set]):
        """
        在内存中直接对 self.processed_documents 的 'structured_data' 进行剪枝。
        1. 移除纯局部的关系。
        2. 移除因关系被剪除而变得孤立的节点。
        """
        self._display_stage_header("内存中进行稀疏化处理")

        total_rels_before = 0
        total_rels_after = 0

        for doc in self.processed_documents:
            if "structured_data" not in doc:
                continue

            for chunk_data in doc["structured_data"]:
                if "relationships" not in chunk_data or "nodes" not in chunk_data:
                    continue

                original_relationships = chunk_data["relationships"]
                total_rels_before += len(original_relationships)

                # 1. 剪枝关系
                kept_relationships = []
                for rel in original_relationships:
                    subject_chunks = provenance_map.get(rel["head"])
                    object_chunks = provenance_map.get(rel["tail"])

                    # 条件1: 两个实体的出处集合不同 (原始的远程关系定义)
                    is_remote_relation = (subject_chunks != object_chunks)

                    # 条件2: 出处集合相同，但它们在多个(>1) chunk中共同出现 (你提出的核心关联)
                    is_core_association = (subject_chunks == object_chunks and len(subject_chunks) > 1)

                    if is_remote_relation or is_core_association:
                        kept_relationships.append(rel)

                total_rels_after += len(kept_relationships)
                chunk_data["relationships"] = kept_relationships  # 更新为剪枝后的关系

                # 2. 剪枝节点：只保留那些仍在剩余关系中被引用的节点
                if not kept_relationships:
                    chunk_data["nodes"] = {}  # 如果没有关系了，就删除所有节点
                else:
                    active_entities = set()
                    for rel in kept_relationships:
                        active_entities.add(rel["head"])
                        active_entities.add(rel["tail"])

                    original_nodes = chunk_data["nodes"]
                    kept_nodes = {
                        node_id: properties
                        for node_id, properties in original_nodes.items()
                        if node_id in active_entities
                    }
                    chunk_data["nodes"] = kept_nodes  # 更新为剪枝后的节点

        # 打印统计信息
        pruned_count = total_rels_before - total_rels_after
        sparsity_ratio = (pruned_count / total_rels_before * 100) if total_rels_before > 0 else 0.0
        # ... (此处可以加入之前版本的美观的 rich.Table 输出) ...
        self.console.print(
            f"[green]内存稀疏化处理完成。剪除 {pruned_count} 条关系，稀疏率: {sparsity_ratio:.2f}%[/green]")

    def _prune_triples_in_memory_2hop(self, provenance_map: Dict[str, Set[str]]):
        """
        在内存中基于“桥梁实体”策略对'structured_data'进行剪枝。
        1. 识别所有“桥梁实体”（出现在多个chunk中的实体）。
        2. 识别所有与桥梁实体直接相关的“一跳关系”和“一跳实体”。
        3. 识别所有连接“一跳实体”的“二跳关系”。
        4. 仅保留“一跳关系”和“二跳关系”，并剪除因此产生的孤立节点。
        """
        self._display_stage_header("内存中进行稀疏化处理 (桥梁实体策略)")

        total_rels_before = 0
        total_rels_after = 0

        # --- 步骤 1: 识别全局的“桥梁实体” ---
        # 这是一个一次性的、跨所有文档的操作，效率更高
        bridge_entities = {
            entity for entity, chunks in provenance_map.items() if len(chunks) > 1
        }
        self.console.print(f"在 {len(provenance_map)} 个实体中，识别出 {len(bridge_entities)} 个桥梁实体。")

        for doc in self.processed_documents:
            if "structured_data" not in doc:
                continue

            for chunk_data in doc["structured_data"]:
                if "relationships" not in chunk_data or "nodes" not in chunk_data:
                    continue

                original_relationships = chunk_data["relationships"]
                total_rels_before += len(original_relationships)

                if not original_relationships:
                    chunk_data["nodes"] = {}
                    continue

                # --- 步骤 2: 识别此chunk内的“一跳关系”和“一跳实体” ---
                one_hop_relationships = []
                one_hop_entities = set()

                for rel in original_relationships:
                    # 只要关系的主体或客体是桥梁实体，就保留这个关系
                    if rel["head"] in bridge_entities or rel["tail"] in bridge_entities:
                        one_hop_relationships.append(rel)
                        # 将这个关系中的两个实体都标记为“一跳实体”
                        one_hop_entities.add(rel["head"])
                        one_hop_entities.add(rel["tail"])

                # --- 步骤 3: 识别此chunk内的“二跳关系” ---
                # 二跳关系是指连接了一个“一跳实体”的关系
                two_hop_relationships = []
                for rel in original_relationships:
                    # 如果一个关系不在一跳关系中 (避免重复)
                    # 并且它的一端至少是“一跳实体”
                    if rel not in one_hop_relationships and (rel["head"] in one_hop_entities or rel["tail"] in one_hop_entities):
                        two_hop_relationships.append(rel)

                # --- 步骤 4: 合并并更新数据 ---
                kept_relationships = one_hop_relationships + two_hop_relationships
                total_rels_after += len(kept_relationships)
                chunk_data["relationships"] = kept_relationships

                # 剪枝节点：只保留那些仍在最终关系中被引用的节点
                if not kept_relationships:
                    chunk_data["nodes"] = {}
                else:
                    active_entities = set()
                    for rel in kept_relationships:
                        active_entities.add(rel["head"])
                        active_entities.add(rel["tail"])

                    original_nodes = chunk_data["nodes"]
                    kept_nodes = {
                        node_id: properties
                        for node_id, properties in original_nodes.items()
                        if node_id in active_entities
                    }
                    chunk_data["nodes"] = kept_nodes

        # 打印统计信息
        pruned_count = total_rels_before - total_rels_after
        sparsity_ratio = (pruned_count / total_rels_before * 100) if total_rels_before > 0 else 0.0
        # ... 此处可以加入之前的美观的 rich.Table 输出 ...
        self.console.print(
            f"[green]内存稀疏化处理完成。剪除 {pruned_count} 条关系，稀疏率: {sparsity_ratio:.2f}%[/green]")

    def _prune_to_bridge_entities(self, provenance_map: Dict[str, Set[str]]):
        """
        在内存中进行稀疏化处理，保留所有桥梁实体节点，但只保留桥梁实体之间的关系。
        此版本采用最高效的单次遍历逻辑。
        """
        self._display_stage_header("内存稀疏化处理 (保留所有桥梁实体节点 - 高效版)")

        # (假设 chunk_to_doc_map 已用于准确识别跨文档的桥梁实体)
        bridge_entities = {
            entity for entity, chunks in provenance_map.items() if len(chunks) > 1
        }
        self.console.print(f"在 {len(provenance_map)} 个实体中，识别出 {len(bridge_entities)} 个桥梁实体。")

        total_rels_before = 0
        total_rels_after = 0

        for doc in self.processed_documents:
            if "structured_data" not in doc:
                continue

            # 在这个循环内完成所有剪枝操作
            for chunk_data in doc["structured_data"]:
                # --- 步骤 1: 过滤关系 ---
                original_relationships = chunk_data.get("relationships", [])
                total_rels_before += len(original_relationships)

                kept_relationships = []
                if original_relationships:
                    for rel in original_relationships:
                        if rel["head"] in bridge_entities and rel["tail"] in bridge_entities:
                            kept_relationships.append(rel)

                total_rels_after += len(kept_relationships)
                chunk_data["relationships"] = kept_relationships

                # --- 步骤 2: 过滤节点 (一步到位) ---
                original_nodes = chunk_data.get("nodes", {})

                # 关键修改：直接在此处根据全局 bridge_entities 集合进行剪枝
                kept_nodes = {
                    node_id: properties
                    for node_id, properties in original_nodes.items()
                    if node_id in bridge_entities
                }
                chunk_data["nodes"] = kept_nodes

        # --- 打印统计信息 (逻辑不变) ---
        pruned_count = total_rels_before - total_rels_after
        sparsity_ratio = (pruned_count / total_rels_before * 100) if total_rels_before > 0 else 0.0
        self.console.print(
            f"[green]内存稀疏化处理完成。剪除 {pruned_count} 条关系，稀疏率: {sparsity_ratio:.2f}%[/green]")

    def build_base_graph(self) -> List:
        """
        构建基础知识图谱
        
        Returns:
            List: 处理后的文件内容列表，包含文件名、原文、分块和处理结果
        """
        self._display_stage_header("构建基础知识图谱")

        try:
            # 1. 处理文件（读取和分块）
            process_start = time.time()
            with self._create_progress() as progress:
                task = progress.add_task("[cyan]处理文件...", total=1)

                # # 使用DocumentProcessor处理文件
                # self.processed_documents = self.document_processor.process_directory()

                # 使用DatasetProcess处理数据集
                self.processed_documents = self.dataset_processor.process_dataset()
                progress.update(task, completed=1)

                # 显示文件信息
                table = Table(title="文件信息")
                table.add_column("文件名")
                table.add_column("类型", style="cyan")
                table.add_column("内容长度", justify="right")
                table.add_column("分块数量", justify="right")

                for doc in self.processed_documents:
                    file_type = self.dataset_processor.get_extension_type(doc["extension"])
                    chunks_count = doc.get("chunk_count", 0)
                    table.add_row(
                        doc["filename"],
                        file_type,
                        str(doc["content_length"]),
                        str(chunks_count)
                    )
                self.console.print(table)

            self.performance_stats["文件处理"] = time.time() - process_start

            # 显示分块统计
            total_chunks = sum(doc.get("chunk_count", 0) for doc in self.processed_documents)
            total_length = sum(doc["content_length"] for doc in self.processed_documents)
            avg_chunk_size = sum(sum(doc.get("chunk_lengths", [0])) for doc in
                                 self.processed_documents) / total_chunks if total_chunks else 0

            self.console.print(f"[blue]共处理 {len(self.processed_documents)} 个文件，总计 {total_length} 字符[/blue]")
            self.console.print(f"[blue]共生成 {total_chunks} 个文本块，平均每块 {avg_chunk_size:.1f} 字符[/blue]")

            # 3. 构建图结构
            struct_start = time.time()
            with self._create_progress() as progress:
                task = progress.add_task("[cyan]构建图结构...", total=3)

                # 清空并创建Document节点
                self.struct_builder.clear_database()
                for doc in self.processed_documents:
                    if "chunks" in doc:  # 只处理文档
                        self.struct_builder.create_document(
                            type="local",
                            uri=str(DATASET_DIR),
                            file_name=doc["filename"],
                            domain=theme
                        )
                progress.advance(task)

                # 创建Chunk节点和关系 - 优化：使用并行处理大文件
                for doc in self.processed_documents:
                    if "chunks" in doc:  # 只处理成功分块的文档
                        # 根据chunks数量选择处理方法
                        chunks = doc["chunks"]
                        title = doc["title"]
                        if doc.get("chunk_count", 0) > 9:
                            # 对于大文件使用并行处理
                            result = self.struct_builder.parallel_process_chunks(
                                doc["filename"],
                                chunks,
                                title,
                                max_workers=os.cpu_count() or 4
                            )
                        else:
                            # 对于小文件使用标准批处理
                            result = self.struct_builder.create_relation_between_chunks(
                                doc["filename"],
                                chunks,
                                title
                            )
                        doc["graph_result"] = result
                progress.advance(task)
                progress.advance(task)

            self.performance_stats["图结构构建"] = time.time() - struct_start

            # 4. 提取实体和关系
            extract_start = time.time()
            with self._create_progress() as progress:
                total_chunks = sum(doc.get("chunk_count", 0) for doc in self.processed_documents)
                task = progress.add_task("[cyan]提取实体和关系...", total=total_chunks)

                def progress_callback(chunk_index):
                    progress.advance(task)

                # 准备处理的数据格式
                file_contents_format = []
                for doc in self.processed_documents:
                    if "chunks" in doc:
                        file_contents_format.append([
                            doc["filename"],
                            doc["content"],
                            doc["chunks"]
                        ])
                self._display_stage_header(f"总chunk数为：{total_chunks}")
                # 根据数据集大小选择处理方法
                if total_chunks > 100000000000000:
                    # 对于大型数据集使用批处理模式
                    self._display_stage_header("正在使用批处理模式")
                    processed_file_contents = self.entity_extractor.process_chunks_batch(
                        file_contents_format,
                        progress_callback
                    )
                else:
                    # 对于小型数据集使用标准并行处理
                    self._display_stage_header("正在使用并行处理模式")
                    processed_file_contents = self.entity_extractor.process_chunks(
                        file_contents_format,
                        progress_callback
                    )

                # 将处理结果合并回文档数据
                file_content_map = {}
                for processed_file in processed_file_contents:
                    if len(processed_file) >= 4:  # 确保有足够的元素
                        filename = processed_file[0]
                        entity_data = processed_file[3]
                        file_content_map[filename] = entity_data

                # 使用映射将结果放回到原始文档中
                for doc in self.processed_documents:
                    if "chunks" in doc:
                        filename = doc["filename"]
                        if filename in file_content_map:
                            doc["entity_data"] = file_content_map[filename]
                        else:
                            self.console.print(f"[yellow]警告: 文件 {filename} 的实体抽取结果未找到[/yellow]")

            self.performance_stats["实体抽取"] = time.time() - extract_start

            # 输出缓存统计
            cache_hits = getattr(self.entity_extractor, 'cache_hits', 0)
            cache_misses = getattr(self.entity_extractor, 'cache_misses', 0)
            total_requests = cache_hits + cache_misses
            cache_rate = (cache_hits / total_requests * 100) if total_requests > 0 else 0

            self.console.print(f"[blue]LLM调用缓存命中率: {cache_rate:.1f}% ({cache_hits}/{total_requests})[/blue]")

            # 5. 预解析实体关系字符串和合并相似实体
            self._pre_parse_entity_data()
            self._merge_local_entities()

            # 6. 内存稀疏化处理
            prune_start = time.time()
            # 6.1 构建全局实体出处地图s
            entity_provenance_map = self._build_entity_provenance_map()

            # 6.2 基于map在内存中剪除局部三元组  1跳 2跳 仅桥实体
            # self._prune_triples_in_memory(entity_provenance_map)
            # self._prune_triples_in_memory_2hop(entity_provenance_map)
            self._prune_to_bridge_entities(entity_provenance_map)

            # self.performance_stats["内存稀疏化"] = time.time() - prune_start

            # 7. 构建2跳关系或超边
            #todo

            # 8. 写入数据库 (现在写入的是稀疏数据)
            write_start = time.time()
            with self._create_progress() as progress:
                task = progress.add_task("[cyan]写入数据库...", total=1)

                graph_writer_data = []
                for doc in self.processed_documents:
                    if "structured_data" in doc:  # 使用新的键
                        graph_writer_data.append([
                            doc["filename"],
                            doc["content"],
                            doc["chunks"],
                            doc.get("graph_result", []),
                            doc.get("structured_data", []),  # 传入稀疏化的结构化数据
                        ])

                # 使用优化的GraphWriter
                graph_writer = GraphWriter(
                    self.graph,
                    batch_size=50,
                    max_workers=os.cpu_count() or 4
                )
                # 写入数据库
                graph_writer.process_and_write_graph_documents(graph_writer_data)
                progress.update(task, completed=1)

            self.performance_stats["写入数据库"] = time.time() - write_start

            self.console.print("[green]基础知识图谱构建完成[/green]")

            # 显示性能统计
            performance_table = Table(title="性能统计")
            performance_table.add_column("处理阶段", style="cyan")
            performance_table.add_column("耗时(秒)", justify="right")
            performance_table.add_column("占比(%)", justify="right")

            total_time = sum(self.performance_stats.values())
            for stage, elapsed in self.performance_stats.items():
                percentage = (elapsed / total_time * 100) if total_time > 0 else 0
                performance_table.add_row(stage, f"{elapsed:.2f}", f"{percentage:.1f}")

            performance_table.add_row("总计", f"{total_time:.2f}", "100.0", style="bold")
            self.console.print(performance_table)

            # 返回处理好的文档列表
            file_contents_compat = []
            for doc in self.processed_documents:
                if "chunks" in doc:
                    content_list = [
                        doc["filename"],
                        doc["content"],
                        doc["chunks"]
                    ]
                    if "entity_data" in doc:
                        content_list.append(doc["entity_data"])
                    file_contents_compat.append(content_list)

            return file_contents_compat

        except Exception as e:
            self.console.print(f"[red]基础图谱构建失败: {str(e)}[/red]")
            raise

    def process(self):
        """执行知识图谱构建流程"""
        try:
            # 记录开始时间
            self.start_time = time.time()

            # 显示系统资源信息
            cpu_count = os.cpu_count() or "未知"
            memory_gb = psutil.virtual_memory().total / (1024 * 1024 * 1024)

            system_info = f"系统信息: CPU核心数 {cpu_count}, 内存 {memory_gb:.1f}GB"
            self.console.print(f"[blue]{system_info}[/blue]")

            # 显示开始面板
            start_text = Text("开始知识图谱构建流程", style="bold cyan")
            self.console.print(Panel(start_text, border_style="cyan"))

            # 构建基础图谱（文档-块-实体-关系）（不包括总结和社区）
            result = self.build_base_graph()

            # 记录结束时间
            self.end_time = time.time()
            elapsed_time = self.end_time - self.start_time

            # 显示完成面板
            success_text = Text("知识图谱构建流程完成", style="bold green")
            self.console.print(Panel(success_text, border_style="green"))

            # 显示总耗时信息
            self.console.print(f"[bold green]总耗时：{self._format_time(elapsed_time)}[/bold green]")

            return result

        except Exception as e:
            # 记录结束时间（即使出错）
            self.end_time = time.time()
            if self.start_time is not None:
                elapsed_time = self.end_time - self.start_time
                self.console.print(f"[bold yellow]中断前耗时：{self._format_time(elapsed_time)}[/bold yellow]")

            error_text = Text(f"构建过程中出现错误: {str(e)}", style="bold red")
            self.console.print(Panel(error_text, border_style="red"))
            raise


if __name__ == "__main__":
    try:
        builder = KnowledgeGraphBuilder()
        builder.process()
    except Exception as e:
        console = Console()
        console.print(f"[red]执行过程中出现错误: {str(e)}[/red]")
