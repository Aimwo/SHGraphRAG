import os

from datasets import load_dataset

from config.settings import CHUNK_SIZE, OVERLAP
from processor.text_chunker import ChineseTextChunker


class DatasetProcessor:
    """
        数据集处理器，用于段落的分块和向量操作等功能
    """

    def __init__(self, dataset_path: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP):
        """
        初始化文档处理器

        Args:
            directory_path: 文件目录路径
        """
        self.dataset_path = dataset_path
        self.chunker = ChineseTextChunker(chunk_size, overlap)
        self.dataset = load_dataset("parquet", data_files=str(self.dataset_path))["train"].select(range(500))

    def process_data(self, data):
        file_ext = os.path.splitext(self.dataset_path)[1]
        content = ""
        for sentences in data["context"]["sentences"]:
            for sentence in sentences:
                content += sentence
                content += "\n"

        # 创建文件处理结果字典
        file_result = {
            "filepath": str(self.dataset_path),  # 相对路径
            "filename": os.path.basename(self.dataset_path) + "_" + data["id"],  # 仅文件名
            "extension": file_ext,
            "content": content,
            "content_length": len(content),
            "question_id": data["id"],
            "title": data["context"]["title"],
            "chunks": None
        }

        # 对文本内容进行分块
        try:
            chunks = []
            for title, sentences in zip(data["context"]["title"], data["context"]["sentences"]):
                # chunk = self.chunker.chunk_text(" ".join(sentences))
                chunk = "\n".join(sentences)
                chunk = title + "\n" + chunk
                chunks.append(chunk)
            file_result["chunks"] = chunks
            file_result["chunk_count"] = len(chunks)

            # 计算每个块的长度
            # chunk_lengths = [len(''.join(chunk)) for chunk in chunks]
            chunk_lengths = [len(''.join(chunk)) for chunk in chunks]
            file_result["chunk_lengths"] = chunk_lengths
            file_result["average_chunk_length"] = sum(chunk_lengths) / len(
                chunk_lengths) if chunk_lengths else 0

        except Exception as e:
            file_result["chunk_error"] = str(e)
            print(f"分块错误 ({self.dataset_path}): {str(e)}")

        return file_result

    def process_dataset(self):
        # 打印调试信息
        print(f"DatasetProcessor找到的测试数据数量: {len(self.dataset)}")
        if len(self.dataset) > 0:
            print(f"文件类型: {os.path.splitext(self.dataset_path)}")
        original_columns = self.dataset.column_names
        # 使用 map 方法并行处理数据集
        results = self.dataset.map(self.process_data,remove_columns=original_columns, cache_file_name=None)  # num_proc 设置并行处理的进程数
        #
        return results.to_pandas().to_dict(orient="records")

    def get_extension_type(self, extension: str) -> str:
        """
        获取文件扩展名对应的文档类型

        Args:
            extension: 文件扩展名（包括'.'，如'.pdf'）

        Returns:
            str: 文档类型描述
        """
        extension_types = {
            '.txt': '文本文件',
            '.pdf': 'PDF文档',
            '.md': 'Markdown文档',
            '.doc': 'Word文档',
            '.docx': 'Word文档',
            '.csv': 'CSV数据文件',
            '.json': 'JSON数据文件',
            '.yaml': 'YAML配置文件',
            '.parquet': 'parquet数据集',
        }

        return extension_types.get(extension.lower(), '未知类型')
