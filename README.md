# Financial-Insight-RAG: 本地财经财报智能分析系统

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-LangChain-green.svg)](https://python.langchain.com/)
[![VectorDB](https://img.shields.io/badge/VectorDB-Chroma-red.svg)](https://www.trychroma.com/)
[![LLM](https://img.shields.io/badge/Local_LLM-Qwen2--1.5B-purple.svg)](https://ollama.com/)

## 📝 项目简介
本项目是一个基于 **RAG (Retrieval-Augmented Generation)** 架构的本地财经问答系统。针对财经报表（如万科 2025 年度报告）中复杂的表格结构和长文本逻辑，实现了从非结构化 PDF 解析到结构化知识库检索，再到大模型专业问答的全流程闭环。

## 🚀 核心技术亮点
- **鲁棒的增量数据流水线**：设计并实现了自动归档（Archiving）机制，确保海量财报 PDF 在处理过程中不重复计算、不污染向量空间。
- **高保真表格解析**：引入 `PDFPlumber` 深度还原财报中的财务报表行列结构，显著提升了模型对营收、现金流等数据的定位精度。
- **专家级 Prompt 工程**：针对轻量化模型（Qwen2-1.5B）设计的结构化提示词，融入了 CPA（注册会计师）角色注入与思维链（CoT）引导，有效解决了模型在复杂数据前的“防御性拒绝”问题。
- **全透明溯源 UI**：基于 Streamlit 构建专业仪表盘，支持 Top-K 检索片段的可视化溯源，确保每一句 AI 回答均有据可查。

## 🛠️ 技术栈
- **语言模型 (LLM)**: Qwen2-1.5B (via Ollama)
- **嵌入模型 (Embedding)**: m3e-small (Moka AI)
- **核心框架**: LangChain (LCEL 语法)
- **向量数据库**: ChromaDB
- **解析工具**: PDFPlumber, RecursiveCharacterTextSplitter
- **前端界面**: Streamlit

## 💻 硬件环境与优化思路
- **测试环境**:
  - **CPU**: Intel Core i7-12700H / AMD Ryzen 7 5800H (及以上性能笔记本 CPU)
  - **内存**: 16GB RAM
  - **GPU**: 无需显卡 (全流程基于 CPU 优化，支持 4-bit 量化加速)
- **优化细节**:
  - **端侧部署优化**: 选用 `Qwen2-1.5B` 小参数模型，通过本地 `Ollama` 部署，在普通笔记本上即可实现秒级响应。
  - **语义检索优化**: 显式设置 `HuggingFaceEmbeddings` 为 `cpu` 运行模式，适配无 GPU 环境，并配合 `normalize_embeddings` 提升检索稳定性。
  - **解析质量优化**: 针对财报特有的多列布局和表格，配置 `PDFPlumberLoader` 进行高保真提取，避免传统解析器导致的语义碎片化。

## 📦 快速开始

### 1. 环境准备
```bash
# 克隆仓库
git clone https://github.com/Lunaira-Y/Financial-Insight-RAG.git
cd Financial-Insight-RAG

# 安装依赖
pip install -r requirements.txt

# 启动 Ollama 并拉取模型
ollama pull qwen2:1.5b
```

### 2. 数据准备
将 PDF 格式的财报文件放入 `data/` 文件夹（注：本项目已通过 `.gitignore` 排除 PDF 原件，以保护个人数据隐私）。

### 3. 运行流水线
```bash
# 步骤 A: 文本切片预处理
python src/section.py

# 步骤 B: 向量化入库 (支持增量更新与自动归档)
python src/vectorize.py
```

### 4. 启动应用
```bash
streamlit run src/app.py
```

## 📂 项目结构
```text
.
├── archive/             # 已处理数据的自动归档目录
├── chroma_db/           # ChromaDB 本地向量数据库 (Git 已忽略)
├── data/                # 原始财报 PDF 存放目录 (Git 已忽略)
├── processed_chunks/    # 预处理生成的 JSON 切片 (Git 已忽略)
├── src/                 # 核心源代码
│   ├── app.py           # Streamlit Web UI
│   ├── rag_chain.py     # RAG 逻辑核心实现
│   ├── section.py       # 文本切片与清洗逻辑
│   └── vectorize.py     # 向量化与归档流程
├── requirements.txt     # 项目依赖列表
└── README.md            # 项目说明文档
```

---
**开发者**: Lunaira  
**致谢**:
感谢 LangChain、Ollama 与 Streamlit 社区提供的优秀开源应用框架；特别感谢 Moka AI 开源的 `m3e` 中文向量模型，以及 Chroma 和 PDFPlumber 在底层数据处理与向量存储上提供的强大技术支撑。
