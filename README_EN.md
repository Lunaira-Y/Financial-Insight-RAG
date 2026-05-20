# Financial-Insight-RAG (Agentic Enhanced RAG System)

<p align="center">
  <a href="README.md">中文</a> | <b>English</b>
</p>

---

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-LangChain-green.svg)](https://python.langchain.com/)
[![VectorDB](https://img.shields.io/badge/VectorDB-Chroma-red.svg)](https://www.trychroma.com/)
[![LLM](https://img.shields.io/badge/Local_LLM-Qwen2--1.5B-purple.svg)](https://ollama.com/)
[![Frontend](https://img.shields.io/badge/Frontend-Next.js_14-black.svg)](https://nextjs.org/)

<p align="center">
  <img src="assets/demo.png" alt="System Demo" width="800">
</p>

## 📝 Introduction
**Financial-Insight-RAG** is a professional-grade Q&A system designed specifically for complex financial reports (e.g., Annual Reports). Built upon a traditional RAG architecture, this project integrates an **Agentic Thinking Chain**, **Lightweight Knowledge Graph Enhancement**, and **HyDE (Hypothetical Document Embeddings)**. It excels at handling data calculations, entity relationships, and cross-lingual retrieval tasks within financial contexts.

## 🚀 Key Optimized Features

### 1. Agentic Financial Calculation Engine
Unlike traditional text matching, the system features a built-in Agent logic. When a user question involves financial calculations (e.g., "Calculate the gross margin for 2024"), the LLM will:
- **Self-Plan**: Identify the required raw accounting data.
- **Precise Extraction**: Extract values from retrieved text chunks and normalize units.
- **Secure Execution**: Automatically generate Python scripts to perform calculations in a sandbox environment, ensuring absolute accuracy.

### 2. Deep Retrieval Enhancement Pipeline
- **Graph Boost**: Utilizes lightweight graph extraction to identify core financial entities (companies, metrics, percentage changes) in queries, dynamically optimizing retrieval weights.
- **HyDE (Hypothetical Document Embeddings)**: Generates hypothetical financial report snippets via LLM to bridge the semantic gap between natural language queries and professional financial terminology.
- **Ensemble Retrieval**: Combines BM25 keyword search with semantic vector search to achieve both precision and deep understanding.

### 3. Modern Full-Stack Architecture
- **Backend (FastAPI)**: High-performance asynchronous API supporting metadata filtering, incremental processing, and offline model loading.
- **Frontend (Next.js 14)**: Built with the latest React Server Components architecture, providing a responsive and fluid streaming interaction experience.
- **Bilingual Routing**: Intelligent recognition of Chinese and English, enabling "cross-lingual analysis" and "expert-level prompt switching."

## 🛠️ Tech Stack
- **LLM**: Qwen2-1.5B (via Ollama)
- **Embedding**: m3e-small (Moka AI, fully offline supported)
- **Frameworks**: LangChain (LCEL), FastAPI
- **Vector Database**: ChromaDB
- **Tokenization**: Jieba (optimized for Chinese financial vocabulary)
- **Frontend**: Next.js (Preferred) / Streamlit (Legacy/Alternative)

## 📂 Project Structure
```text
├── src/
│   ├── api.py            # FastAPI High-performance API
│   ├── app.py            # Streamlit UI (Legacy)
│   ├── graph_engine.py   # Lightweight Entity Extraction Engine
│   ├── retrieval_engine.py # Hybrid Retrieval & HyDE Logic
│   ├── rag_chain.py      # Agentic RAG Core Logic
│   ├── section.py        # High-fidelity PDF Parsing Engine
│   └── vectorize.py      # Vectorization & Incremental Archiving
├── frontend/             # Next.js Modern Frontend
├── data/                 # Raw PDF storage (auto-filtered)
├── models/               # Local model weights (m3e-small)
└── assets/               # Documentation images
```

## 📦 Quick Start

### 1. Backend Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Pull local model
ollama pull qwen2:1.5b
```

### 2. Data Processing
1. Place PDF reports in the `data/` folder.
2. Run parsing: `python src/section.py`
3. Run vectorization: `python src/vectorize.py`

### 3. Start Services
- **Backend API**: `python src/api.py` (Runs on port 8000)
- **Frontend**:
  ```bash
  cd frontend
  npm install
  npm run dev
  ```

## 🤝 Acknowledgments
This project is inspired by the open-source community. Special thanks to:
- **LangChain & Ollama**: For providing the foundation for RAG and local LLM deployment.
- **Moka AI**: For open-sourcing the excellent `m3e` Chinese embedding model.
- **Qwen Team (Alibaba)**: For the impressive Qwen2 series models that excel on edge devices.
- **ChromaDB**: For a simple yet efficient vector storage solution.
- **FlashRank**: For providing technical inspiration during the reranking phase.

---
**Developer**: Lunaira  
**License**: MIT License
