import re
import concurrent.futures
from typing import List, Set
from langchain_community.graphs import Neo4jGraph
from langchain_core.documents import Document
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from typing import List, Dict, Any
from graph.core import connection_manager
from config.settings import BATCH_SIZE as DEFAULT_BATCH_SIZE, MAX_WORKERS as DEFAULT_MAX_WORKERS

class GraphWriter:
    """
    图写入器，负责将提取的实体和关系写入Neo4j图数据库。
    处理实体和关系的解析、转换为GraphDocument，以及批量写入图数据库。
    """
    
    def __init__(self, graph: Neo4jGraph = None, batch_size: int = 50, max_workers: int = 4):
        """
        初始化图写入器
        
        Args:
            graph: Neo4j图数据库对象，如果为None则使用连接管理器获取
            batch_size: 批处理大小
            max_workers: 并行工作线程数
        """
        self.graph = graph or connection_manager.get_connection()
        self.batch_size = batch_size or DEFAULT_BATCH_SIZE
        self.max_workers = max_workers or DEFAULT_MAX_WORKERS
        
        # 节点缓存，用于减少重复节点的创建
        self.node_cache = {}
        
        # 用于跟踪已经处理的节点，减少重复操作
        self.processed_nodes: Set[str] = set()


    def convert_to_graph_document(self, chunk_id: str, input_text: str,
                                  structured_chunk_data: Dict[str, Any]) -> GraphDocument:
        """
        将预解析并已剪枝的结构化字典转换为GraphDocument对象。

        Args:
            chunk_id: 文本块ID
            input_text: 输入文本
            structured_chunk_data: 包含'nodes'和'relationships'的字典

        Returns:
            GraphDocument: 转换后的图文档对象
        """
        nodes = {}
        relationships = []

        try:
            # 1. 创建节点对象
            # 这里的'nodes'已经是剪枝后留下的必要节点
            for node_id, properties in structured_chunk_data.get("nodes", {}).items():
                if node_id not in self.node_cache:
                    self.node_cache[node_id] = Node(
                        id=node_id,
                        type=properties.get("type", "未知"),
                        properties=properties
                    )
                nodes[node_id] = self.node_cache[node_id]

            # 2. 创建关系对象
            # 这里的'relationships'已经是剪枝后留下的远程关系
            for rel_data in structured_chunk_data.get("relationships", []):
                source_id = rel_data["head"]
                target_id = rel_data["tail"]

                # 节点必须存在，因为我们已经同步剪枝了
                if source_id in nodes and target_id in nodes:
                    relationships.append(
                        Relationship(
                            source=nodes[source_id],
                            target=nodes[target_id],
                            type=rel_data["relation"],
                            properties={
                                "description": rel_data["description"],
                                "weight": rel_data["weight"]
                            }
                        )
                    )

        except Exception as e:
            print(f"转换GraphDocument时出错 (Chunk ID: {chunk_id}): {e}")
            # 返回一个包含错误信息的空GraphDocument
            return GraphDocument(nodes=[], relationships=[], source=Document(page_content=input_text,
                                                                             metadata={"chunk_id": chunk_id,
                                                                                       "error": str(e)}))

        return GraphDocument(
            nodes=list(nodes.values()),
            relationships=relationships,
            source=Document(page_content=input_text, metadata={"chunk_id": chunk_id})
        )

    def process_and_write_graph_documents(self, file_contents: List) -> None:
        """
        处理并写入所有文件的GraphDocument对象。
        此版本已更新，以处理预解析和剪枝后的结构化数据。

        Args:
            file_contents: 文件内容列表。
                           其中 file_content[4] 现在是稀疏化的 structured_data 列表。
        """
        all_graph_documents = []
        all_chunk_ids = []

        # 预计算总chunks数，用于预分配列表
        total_chunks = 0
        for file_content in file_contents:
            # file_content[3] 是 'graph_result' (chunks_with_hash)
            if len(file_content) > 3 and file_content[3] is not None:
                total_chunks += len(file_content[3])

        all_graph_documents = [None] * total_chunks
        all_chunk_ids = [None] * total_chunks

        chunk_index = 0
        error_count = 0

        print(f"开始转换并写入 {total_chunks} 个 chunks 的稀疏图数据...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {}

            for file_content in file_contents:
                # 安全地获取数据
                if len(file_content) < 5:
                    continue

                chunks_with_hash = file_content[3]
                # 新变化：这里是结构化、稀疏化的数据，而不再是原始字符串
                sparse_structured_data = file_content[4]

                # 确保数据对齐
                if len(chunks_with_hash) != len(sparse_structured_data):
                    print(f"警告: 文件 {file_content[0]} 的 chunk 数量和结构化数据数量不匹配。")
                    continue

                for i, (chunk_meta, structured_chunk) in enumerate(zip(chunks_with_hash, sparse_structured_data)):
                    # 新变化：第三个参数现在是 structured_chunk (字典)，而不是 result (字符串)
                    future = executor.submit(
                        self.convert_to_graph_document,
                        chunk_meta["chunk_id"],
                        chunk_meta["chunk_doc"].page_content,
                        structured_chunk  # <-- 关键改变在这里
                    )
                    future_to_index[future] = chunk_index
                    chunk_index += 1

            # 收集结果的逻辑完全保持不变，因为它足够健壮
            for future in concurrent.futures.as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    graph_document = future.result()
                    if graph_document and (graph_document.nodes or graph_document.relationships):
                        all_graph_documents[idx] = graph_document
                        all_chunk_ids[idx] = graph_document.source.metadata.get("chunk_id")
                    else:
                        all_graph_documents[idx] = None
                        all_chunk_ids[idx] = None
                except Exception as e:
                    error_count += 1
                    print(f"处理chunk时出错 (已有{error_count}个错误): {e}")
                    all_graph_documents[idx] = None
                    all_chunk_ids[idx] = None

        # 后续处理逻辑也完全保持不变
        all_graph_documents = [doc for doc in all_graph_documents if doc is not None]
        all_chunk_ids = [id for id in all_chunk_ids if id is not None]

        print(f"共转换 {len(all_graph_documents)} 个有效的 GraphDocument，处理期间有 {error_count} 个错误。")

        self._batch_write_graph_documents(all_graph_documents)

        if all_chunk_ids:
            self.merge_chunk_relationships(all_chunk_ids)
    
    def _batch_write_graph_documents(self, documents: List[GraphDocument]) -> None:
        """
        批量写入图文档
        
        Args:
            documents: 图文档列表
        """
        if not documents:
            return
            
        # 增加批处理大小的动态调整
        optimal_batch_size = min(self.batch_size, max(10, len(documents) // 10))
        total_batches = (len(documents) + optimal_batch_size - 1) // optimal_batch_size
        
        print(f"开始批量写入 {len(documents)} 个文档，批次大小: {optimal_batch_size}, 总批次: {total_batches}")
        
        # 批量写入图文档
        for i in range(0, len(documents), optimal_batch_size):
            batch = documents[i:i+optimal_batch_size]
            if batch:
                try:
                    self.graph.add_graph_documents(
                        batch,
                        baseEntityLabel=True,
                        include_source=True
                    )
                    print(f"已写入批次 {i//optimal_batch_size + 1}/{total_batches}")
                except Exception as e:
                    print(f"写入图文档批次时出错: {e}")
                    # 如果批次写入失败，尝试逐个写入以避免整批失败
                    for doc in batch:
                        try:
                            self.graph.add_graph_documents(
                                [doc],
                                baseEntityLabel=True,
                                include_source=True
                            )
                        except Exception as e2:
                            print(f"单个文档写入失败: {e2}")
    
    def merge_chunk_relationships(self, chunk_ids: List[str]) -> None:
        """
        合并Chunk节点与Document节点的关系
        
        Args:
            chunk_ids: 块ID列表
        """
        if not chunk_ids:
            return
        
        # 去除重复的chunk_id以减少操作数量
        unique_chunk_ids = list(set(chunk_ids))
        print(f"开始合并 {len(unique_chunk_ids)} 个唯一chunk关系")
            
        # 动态批处理大小
        optimal_batch_size = min(self.batch_size, max(20, len(unique_chunk_ids) // 5))
        total_batches = (len(unique_chunk_ids) + optimal_batch_size - 1) // optimal_batch_size
        
        print(f"合并关系批次大小: {optimal_batch_size}, 总批次: {total_batches}")
        
        # 分批处理，避免一次性处理过多数据
        for i in range(0, len(unique_chunk_ids), optimal_batch_size):
            batch_chunk_ids = unique_chunk_ids[i:i+optimal_batch_size]
            batch_data = [{"chunk_id": chunk_id} for chunk_id in batch_chunk_ids]
            
            try:
                # 使用原始的查询，确保兼容性
                merge_query = """
                    UNWIND $batch_data AS data
                    MATCH (c:`__Chunk__` {id: data.chunk_id}), (d:Document{chunk_id:data.chunk_id})
                    WITH c, d
                    MATCH (d)-[r:MENTIONS]->(e)
                    MERGE (c)-[newR:MENTIONS]->(e)
                    ON CREATE SET newR += properties(r)
                    DETACH DELETE d
                """
                
                self.graph.query(merge_query, params={"batch_data": batch_data})
                print(f"已处理合并关系批次 {i//optimal_batch_size + 1}/{total_batches}")
            except Exception as e:
                print(f"合并关系批次时出错: {e}")
                # 如果批处理失败，尝试逐个处理
                for chunk_id in batch_chunk_ids:
                    try:
                        single_query = """
                            MATCH (c:`__Chunk__` {id: $chunk_id}), (d:Document{chunk_id:$chunk_id})
                            WITH c, d
                            MATCH (d)-[r:MENTIONS]->(e)
                            MERGE (c)-[newR:MENTIONS]->(e)
                            ON CREATE SET newR += properties(r)
                            DETACH DELETE d
                        """
                        self.graph.query(single_query, params={"chunk_id": chunk_id})
                    except Exception as e2:
                        print(f"处理单个chunk关系时出错: {e2}")