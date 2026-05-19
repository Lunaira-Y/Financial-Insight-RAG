import os
import jieba
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
try:
    from langchain.retrievers import EnsembleRetriever
except (ImportError, ModuleNotFoundError):
    try:
        from langchain_community.retrievers import EnsembleRetriever
    except (ImportError, ModuleNotFoundError):
        from langchain_classic.retrievers import EnsembleRetriever

# 自定义中文分词器
def jieba_tokenizer(text):
    return list(jieba.cut(text))

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

try:
    from src.graph_engine import LightweightGraph
except:
    from graph_engine import LightweightGraph

class ProfessionalRetriever:
    def __init__(self, vectorstore, documents):
        self.vectorstore = vectorstore
        self.documents = documents
        
        # ... (前面的初始化保持不变)
        self.bm25_retriever = BM25Retriever.from_documents(
            documents, 
            preprocess_func=jieba_tokenizer
        )
        self.bm25_retriever.k = 10
        self.vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.vector_retriever],
            weights=[0.3, 0.7]
        )
        self.retrieval_llm = ChatOllama(model="qwen2:1.5b", temperature=0.1)
        
        # ✨ 新增：加载轻量级图谱
        self.graph = LightweightGraph()
        # 注意：实际项目中这里应从持久化文件加载
        # self.graph.build_from_chunks(...)

    def _boost_entities(self, query):
        """识别查询中的实体，用于增强检索权重"""
        entities = self.graph.extract_entities(query)
        if entities:
            # print(f"🔍 [Graph] 识别到实体，正在增强权重: {[e['name'] for e in entities]}")
            return " ".join([e["name"] for e in entities]) + " " + query
        return query

    def _generate_hyde_query(self, query):
        """生成假设性文档 (HyDE) 以增强语义检索"""
        hyde_prompt = ChatPromptTemplate.from_template("""你是一位资深财经分析师。请针对以下问题，根据你的专业知识简要回答一个可能的答案片段。
这个答案不需要完全准确，但其术语和风格应符合财报规范，用于帮助在海量文档中进行语义匹配。

问题：{question}
可能的财报描述：""")
        
        chain = hyde_prompt | self.retrieval_llm | StrOutputParser()
        try:
            hyde_answer = chain.invoke({"question": query})
            return f"{query}\n{hyde_answer}"
        except:
            return query

    def retrieve(self, query, top_n=5, metadata_filter=None):
        """增强型检索：Graph Boost + HyDE + Ensemble"""
        
        # 1. Graph Boost
        boosted_query = self._boost_entities(query)
        
        # 2. HyDE 增强
        enhanced_query = self._generate_hyde_query(boosted_query)
        
        # 3. 混合检索
        if metadata_filter:
            self.vector_retriever.search_kwargs["filter"] = metadata_filter
        
        initial_docs = self.ensemble_retriever.invoke(enhanced_query)
        return initial_docs[:top_n]

def get_retrieval_engine(vectorstore):
    """便捷工厂函数：从向量库中提取所有文档并初始化引擎"""
    all_docs = vectorstore.get()
    if not all_docs["documents"]:
        print("⚠️ 警告：向量库为空，无法初始化检索引擎。请先运行数据预处理和入库脚本。")
        return None
        
    from langchain_core.documents import Document
    documents = [
        Document(page_content=content, metadata=metadata) 
        for content, metadata in zip(all_docs["documents"], all_docs["metadatas"])
    ]
    return ProfessionalRetriever(vectorstore, documents)
