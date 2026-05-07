import os
import json
import shutil  # ✨ 引入用于移动文件的标准库
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
# ✨ 优化点：使用最新版库的导入方式，消除之前的 DeprecationWarning
from langchain_huggingface import HuggingFaceEmbeddings

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def build_vector_database():
    # ---------------------------------------------------------
    # 1. 动态路径解析与归档目录设置
    # ---------------------------------------------------------
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_script_dir)

    input_dir = os.path.join(project_root, "processed_chunks")
    persist_dir = os.path.join(project_root, "chroma_db")
    # ✨ 优化点：设定归档目录
    archive_dir = os.path.join(project_root, "archive", "chunks")

    if not os.path.exists(input_dir):
        print(f"❌ 错误：找不到切片文件夹！请先运行第二阶段代码。")
        return

    # 确保归档文件夹存在
    os.makedirs(archive_dir, exist_ok=True)

    # ---------------------------------------------------------
    # 2. 读取待处理的 JSON 切片
    # ---------------------------------------------------------
    documents = []
    json_files_to_archive = []  # 用于记录本次成功读取了哪些文件

    print("📂 正在检查是否有新的 JSON 切片数据...")
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(input_dir, filename)
            json_files_to_archive.append(file_path)  # 记录路径备用

            with open(file_path, "r", encoding="utf-8") as f:
                chunks_data = json.load(f)
                for item in chunks_data:
                    doc = Document(
                        page_content=item["content"],
                        metadata=item["metadata"]
                    )
                    documents.append(doc)

    # ✨ 优化点：如果没有新数据，直接退出，不唤醒向量模型
    if not documents:
        print("⏭️ 暂无新的切片数据，ChromaDB 向量库无需更新。")
        return

    print(f"  -> 发现新数据，成功加载了 {len(documents)} 个文本块！")

    # ---------------------------------------------------------
    # 3. 初始化 Embedding 模型并入库
    # ---------------------------------------------------------
    print("\n🧠 正在加载 m3e 向量模型...")
    embeddings = HuggingFaceEmbeddings(
        model_name="moka-ai/m3e-small",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    print(f"\n🧱 正在将新文本块转化为向量并追加存入 ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    print(f"🎉 向量数据入库成功！")

    # ---------------------------------------------------------
    # 4. ✨ 流水线归档操作 ✨
    # ---------------------------------------------------------
    print("\n📦 正在进行数据归档...")
    for file_path in json_files_to_archive:
        filename = os.path.basename(file_path)
        archive_path = os.path.join(archive_dir, filename)

        # 将处理完的 JSON 文件移动到 archive 文件夹
        shutil.move(file_path, archive_path)
        print(f"  -> 已归档至: {archive_path}")

    print(f"\n💾 数据库持久化完成，全部增量流水线执行完毕！")


if __name__ == "__main__":
    build_vector_database()