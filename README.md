# Financial-Insight-RAG

<p align="center">
  <b>中文</b> | <a href="README_EN.md">English</a>
</p>

---

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-LangChain-green.svg)](https://python.langchain.com/)
[![VectorDB](https://img.shields.io/badge/VectorDB-Chroma-red.svg)](https://www.trychroma.com/)
[![LLM](https://img.shields.io/badge/Local_LLM-Qwen2--1.5B-purple.svg)](https://ollama.com/)

## 📝 项目简介
本项目是一个基于 **RAG (Retrieval-Augmented Generation)** 架构的本地财经问答系统。针对财经报表（如年度报告）中复杂的表格结构和长文本逻辑，实现了从非结构化 PDF 解析到结构化知识库检索，再到大模型专业问答的全流程闭环。

## 🚀 核心技术特性
- **双语动态 Prompt 路由**：内置语言检测，支持中英双语提问。系统会自动识别语种并路由至对应的专家级 Prompt 模板，实现跨语言的财务分析（如“中文财报，英文回答”）。
- **基于元数据的公司筛选 (Metadata Filtering)**：UI 集成动态公司选择器，利用 ChromaDB 元数据过滤机制实现精准检索，有效避免多份财报间的数据干扰。
- **高鲁棒性增量流水线**：
    - **增量处理**：自动识别已解析文件，避免重复计算。
    - **自动归档**：JSON 切片在入库后自动移动至 `archive/` 目录，确保数据流向清晰。
- **全透明溯源 UI**：基于 Streamlit 构建，实时展示检索到的 Top-K 原始片段、来源文件名及页码。

## 🛠️ 技术栈
- **语言模型 (LLM)**: Qwen2-1.5B (via Ollama)
- **嵌入模型 (Embedding)**: m3e-small (Moka AI)
- **核心框架**: LangChain (LCEL 语法)
- **向量数据库**: ChromaDB
- **解析工具**: PDFPlumber
- **前端界面**: Streamlit

## 💻 硬件环境与优化思路
- **端侧部署**: 全链路支持 **CPU 推理**，普通笔记本（16GB RAM）即可实现流畅运行。
- **性能优化**: 针对财务场景设计的 4-bit 量化适配，以及显式设置的 Embedding CPU 运行模式。

## 📦 快速开始

### 1. 环境准备
```bash
# 克隆仓库
git clone https://github.com/Lunaira-Y/Financial-Insight-RAG-.git
cd Financial-Insight-RAG-

# 安装依赖
pip install -r requirements.txt

# 启动本地模型
ollama pull qwen2:1.5b
```

### 2. 运行流水线
1. 将 PDF 财报放入 `data/` 文件夹。
2. 运行切片预处理：`python src/section.py`
3. 运行向量化入库：`python src/vectorize.py`

### 3. 启动应用
```bash
streamlit run src/app.py
```

## 📂 项目结构
- `src/app.py`: Web 交互界面（含双语路由与元数据过滤逻辑）。
- `src/rag_chain.py`: CLI 命令行交互版。
- `src/section.py`: PDF 高保真解析与切片引擎。
- `src/vectorize.py`: 向量化存储与自动归档流程。

---
**开发者**: Lunaira  
**联系方式**: [Your Email/LinkedIn]
