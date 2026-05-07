import os
import re
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# 语言检测函数
def is_english_query(text):
    if not text:
        return False
    text_without_spaces = re.sub(r'\s+', '', text)
    if not text_without_spaces:
        return False
    english_chars = len(re.findall(r'[a-zA-Z]', text_without_spaces))
    return english_chars / len(text_without_spaces) > 0.6


def run_rag_system():
    # 1. 路径配置
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_script_dir)
    persist_dir = os.path.join(project_root, "chroma_db")

    # 确保与 vectorize.py 的配置参数完全一致
    embeddings = HuggingFaceEmbeddings(
        model_name="moka-ai/m3e-small",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # 2. 加载数据库
    if not os.path.exists(persist_dir):
        print(f"❌ 错误：找不到数据库文件夹 {persist_dir}")
        return

    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

    db_count = vectorstore._collection.count()
    print(f"📊 数据库状态检查：当前库内共有 {db_count} 条文本块。")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 3. 初始化 Qwen2
    llm = ChatOllama(model="qwen2:1.5b", temperature=0)

    # 4. 构建调试版检索函数
    def debug_retriever(query):
        docs = retriever.invoke(query)
        print(f"\n🔍 [DEBUG] 针对问题 '{query}'，检索器找到了 {len(docs)} 个片段：")
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', '未知文件')
            page = doc.metadata.get('page', '未知页码')
            print(f"--- 片段 {i + 1} (来源: {source}, 页码: {page}) ---")
            print(f"{doc.page_content[:100]}...")
        return docs

    # 5. 对话循环与动态路由链条
    print("\n🚀 双语财经 RAG 系统已就绪！(输入 'exit' 退出)")
    while True:
        query = input("\n👤 提问: ")
        if query.strip().lower() in ['exit', 'quit', '退出']:
            break
        if not query.strip():
            continue

        try:
            # 根据语言分配不同的 Prompt
            if is_english_query(query):
                template = """# Role
You are a Senior CPA and Cross-national Financial Analyst with 10 years of experience.

# Task
Answer the user's question purely based on the provided [Context].
The [Context] is in Chinese, but you MUST translate the findings and reply entirely in fluent Business English.

# Rules
1. **Table Recognition**: Consecutive numbers usually represent year-over-year data. Match them with the correct row/column headers.
2. **Accuracy & Translation**: Extract the numbers accurately. Translate financial units correctly (e.g., RMB, Yuan).
3. **Honesty**: If the answer is not in the context, clearly reply: "Sorry, the provided financial snippets do not contain specific data regarding this query." DO NOT hallucinate.

# Context
{context}

# User Question
{question}

# Output Format (in English):
Please output in a professional tone. If providing data, use this format:
- **Metric**:
- **Value**:
- **Source Context**: (Briefly explain where you found this in the context)
"""
            else:
                template = """# Role
你是一位拥有 10 年经验的高级注册会计师（CPA）和跨国财务分析专家。你擅长从复杂的财务报表文本和表格中精准提取关键数据。

# Task
请仅基于提供的【参考资料】用专业的中文回答用户的问题。如果资料涉及多个年份或多项财务指标，请进行准确对比和说明。

# Data Handling Rules (针对表格文本)
1. **表格识别**：参考资料中若出现连续的数字，通常代表不同年度的对比数据。请根据行标题和列标题进行精准匹配。
2. **数值完整性**：提取数值时请保留原始精度，并留意金额单位（如：人民币元、万元）。
3. **诚实原则**：
   - 如果参考资料明确包含答案，请直接给出结果。
   - 如果资料中完全没有提及该信息，请明确回答：“抱歉，当前提供的财报片段中未包含有关该问题的具体数据。”严禁胡乱编造。

# Context (参考资料)
{context}

# User Question
{question}

# Output Format
请以专业、客观的口吻给出答案。如果是数据回答，请尽量采用以下格式：
- **指标名称**：
- **具体数值**：
- **财报依据**：(简述在资料中看到的上下文)
"""
            prompt = ChatPromptTemplate.from_template(template)

            rag_chain = (
                    {"context": debug_retriever, "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
            )

            response = rag_chain.invoke(query)
            print(f"\n🤖 回答: {response}")
        except Exception as e:
            print(f"❌ 出错啦: {e}")


if __name__ == "__main__":
    run_rag_system()