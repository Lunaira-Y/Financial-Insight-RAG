import os

# ✨ 关键网络修复：设置 Hugging Face 国内镜像，解决 client closed 报错问题
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 如果依然报错，可以取消下面这行的注释来强制完全离线运行
# os.environ["HF_HUB_OFFLINE"] = "1"

import streamlit as st
import re
from src.retrieval_engine import get_retrieval_engine
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# ==========================================
# 0. 核心语言检测工具 (动态路由核心)
# ==========================================
def is_english_query(text):
    """
    检测用户提问是否主要为英文。
    如果英文字母占比超过 60%，则判定为纯英文环境。
    """
    if not text:
        return False
    text_without_spaces = re.sub(r'\s+', '', text)
    if not text_without_spaces:
        return False
    english_chars = len(re.findall(r'[a-zA-Z]', text_without_spaces))
    return english_chars / len(text_without_spaces) > 0.6


# ==========================================
# 1. 页面基本配置
# ==========================================
st.set_page_config(
    page_title="Lunaira 财经 RAG 助手",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义 CSS 优化视觉效果
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stChatMessage { border-radius: 15px; margin-bottom: 10px; }
    .sidebar .sidebar-content { background-color: #ffffff; }
    </style>
""", unsafe_allow_html=True)


# ==========================================
# 2. 核心后端逻辑加载 (带缓存)
# ==========================================
@st.cache_resource
def load_rag_backend():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    persist_dir = os.path.join(project_root, "chroma_db")
    model_path = os.path.join(project_root, "models", "m3e-small")

    # ✨ 优化：优先使用本地模型
    if os.path.exists(model_path):
        embedding_model = model_path
    else:
        embedding_model = "moka-ai/m3e-small"

    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    if not os.path.exists(persist_dir):
        return None, None, None, 0

    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    db_count = vectorstore._collection.count()

    # ✨ 升级：初始化专业检索引擎
    retrieval_engine = get_retrieval_engine(vectorstore)

    llm = ChatOllama(model="qwen2:1.5b", temperature=0)

    return vectorstore, retrieval_engine, llm, db_count


vectorstore, retrieval_engine, llm, db_count = load_rag_backend()

# ==========================================
# 3. 侧边栏设计 (Professional Info)
# ==========================================
with st.sidebar:
    st.title("🛡️ 系统控制中心")
    st.subheader("项目：财经文本 RAG 系统")

    with st.expander("📊 数据库状态", expanded=True):
        st.write(f"**知识条目数**: `{db_count}`")
        st.write(f"**底层模型**: `Qwen2-1.5B` (Ollama)")
        st.write(f"**向量模型**: `m3e-small` (HuggingFace)")

    st.divider()
    st.subheader("⚙️ 检索参数调节")
    top_k = st.slider("召回片段数量 (K值)", 1, 10, 5)

    st.divider()
    st.subheader("🏢 公司筛选 (Metadata Filter)")
    # 动态获取 PDF 列表
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_path = os.path.join(project_root, "data")

    pdf_files = []
    if os.path.exists(data_path):
        pdf_files = [f for f in os.listdir(data_path) if f.endswith(".pdf")]

    company_options = ["全部公司"] + pdf_files
    selected_company = st.selectbox("选择目标公司进行精准检索", company_options)

    st.divider()
    st.info("""
    **开发者**: Lunaira  
    **阶段**: [Professional V2]  
    **特性**: 支持高保真表格解析、Jieba 增强混合检索。
    """)

    if st.button("🧹 清除聊天历史"):
        st.session_state.messages = []
        st.rerun()

# ==========================================
# 4. 聊天界面逻辑
# ==========================================
st.title("🏦 智能双语财报分析助手 [Professional V2]")
st.write("已启用：高保真表格解析、Jieba 增强混合检索 (Hybrid Search)。")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt_input := st.chat_input("输入财报问题 / Ask a financial question..."):
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)

    with st.chat_message("assistant"):
        if vectorstore is None:
            st.error("❌ 未检测到 ChromaDB 数据库，请确保已运行阶段 3 代码。")
        else:
            with st.spinner("🔍 正在执行专业混合检索并重排序..."):
                # 构造过滤条件
                metadata_filter = None
                if selected_company != "全部公司":
                    target_path = os.path.join(data_path, selected_company)
                    metadata_filter = {"source": target_path}

                # ✨ 调用专业检索引擎
                source_docs = retrieval_engine.retrieve(
                    prompt_input, 
                    top_n=top_k, 
                    metadata_filter=metadata_filter
                )

                # 提取 Context
                context_text = "\n\n".join([doc.page_content for doc in source_docs])

                # ✨ 核心：动态 Prompt 路由 (保留之前的模板逻辑)
                if is_english_query(prompt_input):
                    template = """# Role
You are a Senior CPA and Cross-national Financial Analyst with 10 years of experience.

# Task
Answer the user's question purely based on the provided [Context].
The [Context] is in Chinese, but you MUST translate the findings and reply entirely in fluent Business English.

# Rules
1. **Table Recognition**: Consecutive numbers usually represent year-over-year data. Match them with the correct row/column headers.
2. **Accuracy & Units**: Extract numbers accurately. Pay close attention to units (e.g., RMB Yuan, Thousand, Million, Billion).
3. **Honesty**: If the answer is not in the context, clearly reply: "Sorry, the provided financial snippets do not contain specific data regarding this query." DO NOT hallucinate.

# Context
{context}

# User Question
{question}

# Output Format (in English):
- **Metric**:
- **Value**:
- **Source Context**: (Briefly explain where you found this in the context)
"""
                else:
                    template = """# Role
你是一位拥有 10 年经验的高级注册会计师（CPA）和跨国财务分析专家。你擅长从复杂的财务报表文本和表格中精准提取关键数据。

# Task
请仅基于提供的【参考资料】用专业的中文回答用户的问题。如果资料涉及多个年份或多项财务指标，请进行准确对比和说明。

# Rules
1. **表格识别**：请仔细比对行标题和列标题。连续数字通常代表不同年度（如 2025 vs 2024）。
2. **数值与单位**：提取数值时请务必留意金额单位（如：人民币元、万元、亿元），并在回答中明确标注。
3. **诚实原则**：如果参考资料中完全没有提及该信息，请回答：“抱歉，当前提供的财报片段中未包含有关该问题的具体数据。”

# Context (参考资料)
{context}

# User Question
{question}

# Output Format
- **指标名称**：
- **具体数值**：
- **财报依据**：(简述在资料中看到的上下文)
"""

                prompt_temp = ChatPromptTemplate.from_template(template)

                # ✨ 升级：使用流式输出提升交互感
                chain = (
                        {"context": lambda x: context_text, "question": RunnablePassthrough()}
                        | prompt_temp
                        | llm
                        | StrOutputParser()
                )

                response_placeholder = st.empty()
                full_response = ""
                
                # 开始流式迭代
                for chunk in chain.stream(prompt_input):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                
                response_placeholder.markdown(full_response)

                with st.expander("📚 查看原始检索证据 (Traceability)"):
                    for i, doc in enumerate(source_docs):
                        st.markdown(f"**【片段 {i + 1}】**")
                        st.caption(
                            f"来源: `{os.path.basename(doc.metadata.get('source', '未知'))}` | 页码: `P{doc.metadata.get('page', '?')}`")
                        st.text_area(f"内容内容-{i}", value=doc.page_content[:300] + "...", height=100,
                                     label_visibility="collapsed")
                        st.divider()

                st.session_state.messages.append({"role": "assistant", "content": full_response})

if not st.session_state.messages:
    st.markdown("""
    ---
    ### 💡 建议提问 / Suggested Queries：
    - 万科 2025 年的资产负债情况如何？
    - What are the main financial risks mentioned in the report?
    - 万科在 2025 年的营业收入变动幅度是多少？
    """)