import os

# ✨ 关键网络修复：设置 Hugging Face 国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

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


try:
    from src.retrieval_engine import get_retrieval_engine
    from src.tools import safe_python_executor
except (ImportError, ModuleNotFoundError):
    from retrieval_engine import get_retrieval_engine
    from tools import safe_python_executor

def run_rag_system():
    # ... (前面的加载逻辑保持不变)
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_script_dir)
    persist_dir = os.path.join(project_root, "chroma_db")
    model_path = os.path.join(project_root, "models", "m3e-small")

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
        print(f"❌ 错误：找不到数据库文件夹 {persist_dir}")
        return

    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    retrieval_engine = get_retrieval_engine(vectorstore)
    llm = ChatOllama(model="qwen2:1.5b", temperature=0)

    # 4. 构建 Agentic 执行逻辑
    def agentic_solver(query):
        # 阶段 A：检索数据
        docs = retrieval_engine.retrieve(query, top_n=5)
        context = "\n\n".join([doc.page_content for doc in docs])

        # 阶段 B：让 LLM 生成计算任务 (如果需要)
        planning_prompt = f"""你是一位专业的财务数据分析师。
请阅读【参考资料】，并根据用户问题提出计算方案。
如果问题涉及计算，请输出一段 Python 代码来执行计算。
代码必须简单且只使用 print() 输出结果。

# 参考资料
{context}

# 用户问题
{query}

# 输出要求
请严格按以下 JSON 格式输出：
{{
  "thought": "你的思考过程",
  "need_calculation": true/false,
  "python_code": "你的代码",
  "extracted_data": "从资料中提取的关键数值和单位"
}}
"""
        # 使用 LLM 生成 JSON
        plan_response = llm.invoke(planning_prompt).content
        
        # 简单解析 JSON (针对小模型可能输出非纯 JSON 的情况进行清洗)
        try:
            import json
            # 尝试提取 ```json ... ``` 块
            json_match = re.search(r'\{.*\}', plan_response, re.DOTALL)
            plan = json.loads(json_match.group(0)) if json_match else {}
        except:
            plan = {}

        calc_result = ""
        if plan.get("need_calculation") and plan.get("python_code"):
            print(f"🛠️ [Agent] 正在执行计算脚本...")
            calc_result = safe_python_executor(plan["python_code"])
            print(f"✅ [Agent] 计算结果: {calc_result}")

        # 阶段 C：最终总结回答
        final_prompt = f"""你是一位高级注册会计师。请根据检索到的资料和计算结果，回答用户的问题。
使用专业的财务口吻，并标注财报依据。

# 检索资料
{context}

# 提取数据
{plan.get('extracted_data', '未提取')}

# 计算验证结果
{calc_result}

# 用户问题
{query}
"""
        return llm.invoke(final_prompt).content

    # 5. 对话循环
    print("\n🚀 Agentic RAG CLI 已就绪！(已启用 Python 计算工具)")
    while True:
        query = input("\n👤 提问: ")
        if query.strip().lower() in ['exit', 'quit', '退出']:
            break
        if not query.strip():
            continue

        try:
            response = agentic_solver(query)
            print(f"\n🤖 回答: {response}")
        except Exception as e:
            print(f"❌ 出错啦: {e}")


if __name__ == "__main__":
    run_rag_system()


if __name__ == "__main__":
    run_rag_system()