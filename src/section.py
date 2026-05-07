import os
import json
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def process_financial_reports_optimized():
    # ---------------------------------------------------------
    # 1. 动态路径解析
    # ---------------------------------------------------------
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_script_dir)

    input_dir = os.path.join(project_root, "data")
    output_dir = os.path.join(project_root, "processed_chunks")

    if not os.path.exists(input_dir):
        print(f"❌ 错误：找不到数据文件夹！\n请确保在以下路径存在文件夹：{input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    # ---------------------------------------------------------
    # 2. 初始化切片器
    # ---------------------------------------------------------
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
    )

    pdf_files = [f for f in os.listdir(input_dir) if f.endswith(".pdf")]

    if not pdf_files:
        print(f"⚠️ 警告：在 {input_dir} 文件夹中没有找到任何 PDF 文件。")
        return

    # ---------------------------------------------------------
    # 3. 核心处理流程：带有增量防重复检查
    # ---------------------------------------------------------
    for filename in pdf_files:
        file_path = os.path.join(input_dir, filename)

        # 提前拼装出预期的输出路径
        output_filename = filename.replace(".pdf", "_chunks.json")
        output_path = os.path.join(output_dir, output_filename)

        # ✨ 优化点：增量防重复检查
        if os.path.exists(output_path):
            print(f"⏭️ 跳过：{filename} 的切片已存在，无需重复处理。")
            continue

        print(f"📄 正在使用 PDFPlumber 深度解析财报: {filename} ...")

        loader = PDFPlumberLoader(file_path)
        documents = loader.load()

        chunks = text_splitter.split_documents(documents)
        print(f"  -> ✂️ 成功切分为 {len(chunks)} 个高质量文本块。")

        output_data = []
        for i, chunk in enumerate(chunks):
            output_data.append({
                "chunk_id": i,
                "content": chunk.page_content,
                "metadata": chunk.metadata
            })

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"  -> 💾 切片已单独保存至: {output_path}\n")

    print("🎉 第二阶段运行完毕！")


if __name__ == "__main__":
    process_financial_reports_optimized()