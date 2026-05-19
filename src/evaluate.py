import os
import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevance, context_precision, context_recall
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from src.retrieval_engine import get_retrieval_engine

def run_evaluation():
    # 1. 环境与模型准备
    embeddings = HuggingFaceEmbeddings(
        model_name="moka-ai/m3e-small",
        model_kwargs={'device': 'cpu'}
    )
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    persist_dir = os.path.join(project_root, "chroma_db")
    
    if not os.path.exists(persist_dir):
        print("❌ 数据库不存在，请先运行 vectorize.py")
        return

    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    retrieval_engine = get_retrieval_engine(vectorstore)
    
    # 评测用 LLM (建议使用能力较强的模型作为裁判，如 Qwen2-7B 或 GPT-4)
    eval_llm = ChatOllama(model="qwen2:1.5b") 

    # 2. 构造测试集 (Benchmark Dataset)
    # 比赛建议准备 10-20 个这类问题
    test_questions = [
        {
            "question": "万科 2024 年的营业收入是多少？",
            "ground_truth": "请根据实际财报填入标准答案，用于计算召回率"
        },
        {
            "question": "报告中提到的主要财务风险有哪些？",
            "ground_truth": "市场风险、信用风险、流动性风险等"
        }
    ]

    # 3. 执行 RAG 流程获取结果
    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }

    print(f"🚀 开始对 {len(test_questions)} 个问题进行自动化评测...")

    for item in test_questions:
        query = item["question"]
        # 检索
        docs = retrieval_engine.retrieve(query, top_n=3)
        contexts = [doc.page_content for doc in docs]
        
        # 生成回答 (简化版 Chain)
        context_str = "\n\n".join(contexts)
        prompt = f"基于以下内容回答问题：\n{context_str}\n\n问题：{query}"
        response = eval_llm.invoke(prompt).content
        
        data["question"].append(query)
        data["answer"].append(response)
        data["contexts"].append(contexts)
        data["ground_truth"].append(item["ground_truth"])

    dataset = Dataset.from_dict(data)

    # 4. 调用 RAGAS 评测
    # 注意：RAGAS 在国内使用可能需要配置代理或镜像
    print("📊 正在计算 RAGAS 指标 (Faithfulness, Relevancy...)...")
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevance,
            context_precision,
            context_recall,
        ],
        llm=eval_llm,
        embeddings=embeddings
    )

    # 5. 输出结果
    print("\n✅ 评测完成！结果如下：")
    print(result)
    
    # 保存结果
    with open(os.path.join(project_root, "evaluation_report.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    print(f"\n报告已保存至 evaluation_report.json")

if __name__ == "__main__":
    run_evaluation()
