# **Sparse Hypergraph RAG: Bridging Structured Knowledge for Multi-Hop QA.**


SHGraphRAG is a hypergraph-based RAG framework that preserves key entities and uses 2-hop reasoning to improve multi-document QA.

## 🏗️ **Architecture**
<img width="1472" height="784" alt="image" src="https://github.com/user-attachments/assets/82fd4468-a74a-4f04-9194-92adcca56872" />

---

## 📦 **Installation**

### **1️⃣ Clone this repository**

```bash
git clone https://github.com/Aimwo/SHGraphRAG.git
cd SHGraphRAG
```

### **2️⃣ Create environment**

```bash
conda create -n SHGraphRAG python=3.12
conda activate SHGraphRAG
```

### **3️⃣ Install dependencies**

```bash
pip install -r requirements.txt
```

---

## 🚀 **Quick Start（快速启动）**

### **1️⃣ Prepare data**

```
dataset/xxx/xxx.parquet
```
下面是一个适合放在 README 的英文环境变量配置说明（简洁、规范）：

---

### 🔧 Environment Configuration

Before running the project, please configure the required environment variables:

```bash
export OPENAI_API_KEY="your_openai_api_key"
export OLLAMA_EMBEDDINGS_MODEL="your_embedding_model_name"
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="your_password"
```

---

如果你想，我还可以补充 `.env` 文件示例或用 `dotenv` 自动加载的版本。

### **Build the Graph**

```bash
# 执行完整构建
python -m build.main

### **3️⃣ Inference**

```bash
python qa/hotpot_evaluate_v1.py
```

---

## 📊 **Results**

<img width="747" height="389" alt="image" src="https://github.com/user-attachments/assets/178fdc64-8952-4bd9-bc04-573cff5efae3" />


