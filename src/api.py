import os
import re
import ssl
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

try:
    from src.retrieval_engine import get_retrieval_engine
except (ImportError, ModuleNotFoundError):
    from retrieval_engine import get_retrieval_engine

# ✨ 关键网络修复：设置 Hugging Face 国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 🛠️ 强制完全离线运行开关 (如果报错，可以尝试取消下一行注释)
# os.environ["HF_HUB_OFFLINE"] = "1"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"

# 🛠️ 强制禁用全局 SSL 验证 (针对 requests 库，解决 FlashRank 等下载问题)
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
original_get = requests.get
def patched_get(*args, **kwargs):
    kwargs['verify'] = False
    return original_get(*args, **kwargs)
requests.get = patched_get

try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

app = FastAPI(title="Financial RAG Professional API")

# 启用 CORS 以支持 Next.js 前端访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 开发环境允许所有来源，生产环境应严格限制
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 核心后端逻辑加载 (全局单例)
# ==========================================
class RAGBackend:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        persist_dir = os.path.join(project_root, "chroma_db")
        
        # ✨ 终极修复：优先检查本地是否存在模型文件夹，实现完全脱网
        local_model_path = os.path.join(project_root, "models", "m3e-small")
        model_id = "moka-ai/m3e-small"
        
        if os.path.exists(local_model_path):
            print(f"🚀 检测到本地模型，正在从 {local_model_path} 加载...")
            model_id = local_model_path
        else:
            print(f"🌐 未检测到本地模型，尝试通过镜像站下载: {model_id}")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_id,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        if not os.path.exists(persist_dir):
            self.vectorstore = None
            self.retrieval_engine = None
        else:
            self.vectorstore = Chroma(persist_directory=persist_dir, embedding_function=self.embeddings)
            self.retrieval_engine = get_retrieval_engine(self.vectorstore)

        self.llm = ChatOllama(model="qwen2:1.5b", temperature=0)

    def is_english_query(self, text: str) -> bool:
        if not text:
            return False
        text_without_spaces = re.sub(r'\s+', '', text)
        if not text_without_spaces:
            return False
        english_chars = len(re.findall(r'[a-zA-Z]', text_without_spaces))
        return english_chars / len(text_without_spaces) > 0.6

backend = RAGBackend()

# ==========================================
# API 数据模型
# ==========================================
class ChatRequest(BaseModel):
    prompt: str
    companies: Optional[List[str]] = ["全部公司"]
    top_k: Optional[int] = 5

try:
    from src.retrieval_engine import get_retrieval_engine
    from src.tools import safe_python_executor
except (ImportError, ModuleNotFoundError):
    from retrieval_engine import get_retrieval_engine
    from tools import safe_python_executor

# ... (前面的 RAGBackend 初始化保持不变)
class ChatResponse(BaseModel):
    answer: str
    thought: Optional[str] = None
    calculation: Optional[str] = None
    sources: List[dict]

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if backend.vectorstore is None:
        raise HTTPException(status_code=500, detail="ChromaDB not found.")

    # 1. 构造多公司过滤条件
    metadata_filter = None
    if request.companies and "全部公司" not in request.companies:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        data_path = os.path.join(project_root, "data")
        filter_list = [{"source": os.path.join(data_path, c)} for company in request.companies]
        metadata_filter = {"$or": filter_list} if len(filter_list) > 1 else filter_list[0]

    # 2. 检索阶段
    source_docs = backend.retrieval_engine.retrieve(request.prompt, top_n=request.top_k, metadata_filter=metadata_filter)
    context_text = ""
    for i, doc in enumerate(source_docs):
        context_text += f"\n[证据 {i+1} | 来源: {doc.metadata.get('filename')} | P{doc.metadata.get('page')}]:\n{doc.page_content}\n"

    # 3. Agent 思考与计算阶段
    planning_prompt = f"""你是一位严谨的财务分析师。请阅读资料并针对问题提出计算方案。
重要：如果资料中缺少数字，请将 need_calculation 设为 false。

# 任务要求
1. 提取原始数据：请注明金额单位（如：亿元）。如果是“1,234.56”，请去掉逗号变为“1234.56”。
2. 编写 Python 代码：
   - 变量命名要清晰（如：rev_2025）。
   - 仅使用 print() 输出最后的结果。
   - **严禁**对 None 或非数字进行运算。

# 参考资料
{context_text}

# 用户问题
{request.prompt}

请严格按以下 JSON 格式输出：
{{
  "thought": "你的分析思路",
  "extracted_data": "提取的具体数值（带单位）",
  "need_calculation": true/false,
  "python_code": "在这里写 Python 代码，确保数字格式正确"
}}
"""
    try:
        plan_raw = backend.llm.invoke(planning_prompt).content
        import json
        json_match = re.search(r'\{.*\}', plan_raw, re.DOTALL)
        plan = json.loads(json_match.group(0)) if json_match else {}
    except:
        plan = {}

    calc_result = ""
    if plan.get("need_calculation") and plan.get("python_code"):
        calc_result = safe_python_executor(plan["python_code"])

    # 4. 最终回答
    final_prompt = f"""你是一位 CPA。请结合资料和计算结果回答问题。
资料：{context_text}
提取数据：{plan.get('extracted_data')}
计算结果：{calc_result}
问题：{request.prompt}
"""
    answer = backend.llm.invoke(final_prompt).content

    return ChatResponse(
        answer=answer,
        thought=plan.get("thought"),
        calculation=calc_result if plan.get("need_calculation") else None,
        sources=[{"filename": d.metadata.get("filename"), "page": d.metadata.get("page"), "content": d.page_content} for d in source_docs]
    )

@app.get("/api/companies")
async def get_companies():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_path = os.path.join(project_root, "data")
    
    if not os.path.exists(data_path):
        return []
    
    return [f for f in os.listdir(data_path) if f.endswith(".pdf")]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
