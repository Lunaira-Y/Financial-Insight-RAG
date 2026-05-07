# Financial-Insight-RAG

<p align="center">
  <a href="README.md">中文</a> | <b>English</b>
</p>

---

English | [中文](README.md)

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-LangChain-green.svg)](https://python.langchain.com/)
[![VectorDB](https://img.shields.io/badge/VectorDB-Chroma-red.svg)](https://www.trychroma.com/)
[![LLM](https://img.shields.io/badge/Local_LLM-Qwen2--1.5B-purple.svg)](https://ollama.com/)

## 📝 Overview
This project is a local financial Q&A system based on the **RAG (Retrieval-Augmented Generation)** architecture. Designed for complex table structures and long-text logic in financial reports (e.g., annual reports), it implements a full closed-loop process from unstructured PDF parsing to structured knowledge base retrieval and professional LLM-based Q&A.

![System Demo Interface](assets/demo.png)

## 🚀 Key Features
- **Bilingual Dynamic Prompt Routing**: Built-in language detection supporting both Chinese and English queries. The system automatically identifies the language and routes to the corresponding expert prompt template, enabling cross-language financial analysis (e.g., "Analyze Chinese reports in English").
- **Metadata-based Company Filtering**: Integrated dynamic company selector in the UI. It leverages ChromaDB's metadata filtering mechanism for precise retrieval, effectively avoiding data interference between different financial reports.
- **Robust Incremental Pipeline**:
    - **Incremental Processing**: Automatically identifies parsed files to avoid redundant computation.
    - **Automatic Archiving**: JSON chunks are moved to the `archive/` directory after indexing, ensuring a clear data flow.
- **Transparent Traceability UI**: Built with Streamlit, it displays the Top-K original retrieved snippets, source filenames, and page numbers in real-time.

## 🛠️ Tech Stack
- **LLM**: Qwen2-1.5B (via Ollama)
- **Embedding**: m3e-small (Moka AI)
- **Framework**: LangChain (LCEL)
- **Vector Database**: ChromaDB
- **Parsing Tool**: PDFPlumber
- **Frontend**: Streamlit

## 💻 Hardware & Optimization
- **Edge Deployment**: Supports **CPU inference** across the entire pipeline. It runs smoothly on a standard laptop (16GB RAM).
- **Optimization**: Specifically designed 4-bit quantization for financial scenarios and explicit CPU mode for Embedding models.

## 📦 Quick Start

### 1. Setup
```bash
# Clone the repository
git clone https://github.com/Lunaira-Y/Financial-Insight-RAG.git
cd Financial-Insight-RAG

# Install dependencies
pip install -r requirements.txt

# Pull local model
ollama pull qwen2:1.5b
```

### 2. Run Pipeline
1. Place PDF reports in the `data/` folder.
2. Run slicing: `python src/section.py`
3. Run vectorization: `python src/vectorize.py`

### 3. Launch App
```bash
streamlit run src/app.py
```

## 📂 Project Structure
- `src/app.py`: Web interface (includes bilingual routing and metadata filtering).
- `src/rag_chain.py`: CLI version for interactive testing.
- `src/section.py`: High-fidelity PDF parsing and slicing engine.
- `src/vectorize.py`: Vector storage and automatic archiving process.

---
**Developer**: Lunaira  
**Acknowledgments**:
Thanks to the LangChain, Ollama, and Streamlit communities for providing excellent open-source application frameworks. Special thanks to Moka AI for open-sourcing the m3e Chinese embedding model, as well as Chroma and PDFPlumber for their powerful technical support in underlying data processing and vector storage.
