import os
import json
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

def table_to_markdown(table):
    """将 pdfplumber 提取的表格列表转换为 Markdown 格式"""
    if not table:
        return ""
    
    # 过滤掉全为空的行
    table = [row for row in table if any(cell is not None and str(cell).strip() != "" for cell in row)]
    if not table:
        return ""

    markdown = ""
    for i, row in enumerate(table):
        # 处理 None 值并转为字符串
        clean_row = [str(cell).replace("\n", " ").strip() if cell is not None else "" for cell in row]
        markdown += "| " + " | ".join(clean_row) + " |\n"
        if i == 0:  # 添加分割线
            markdown += "| " + " | ".join(["---"] * len(clean_row)) + " |\n"
    return markdown

def extract_with_layout_cleaning(file_path):
    """
    高保真解析：自动检测并过滤页眉页脚，增强表格关联
    """
    all_page_content = []
    
    with pdfplumber.open(file_path) as pdf:
        # 1. 预扫描：识别可能的页眉页脚 (在多页中重复出现的行)
        line_stats = {}
        total_pages = len(pdf.pages)
        if total_pages > 5: # 仅对多页文档进行预扫描
            sample_pages = pdf.pages[:3] + pdf.pages[-3:] # 采样前三页和后三页
            for page in sample_pages:
                text = page.extract_text()
                if text:
                    lines = [line.strip() for line in text.split("\n")[:2] + text.split("\n")[-2:]]
                    for line in lines:
                        if len(line) > 2:
                            line_stats[line] = line_stats.get(line, 0) + 1
        
        # ... (前面的噪音检测逻辑保持不变)
        noise_lines = {line for line, count in line_stats.items() if count >= (total_pages * 0.3)}

        for i, page in enumerate(pdf.pages):
            # 1. 提取原始文本
            text = page.extract_text() or ""
            lines = text.split("\n")
            
            # 过滤噪音行
            cleaned_lines = [l for l in lines if l.strip() not in noise_lines]
            cleaned_text = "\n".join(cleaned_lines)
            
            # 2. 提取表格并转化为 Markdown
            tables = page.extract_tables()
            md_tables = ""
            for table in tables:
                md_tables += table_to_markdown(table) + "\n\n"
            
            # 3. ✨ 深度增强：图片/图表 OCR 识别 (Vision-to-Text)
            images = page.images
            vision_text_content = ""
            if images:
                import pytesseract
                from PIL import Image
                import io
                
                print(f"  -> 📸 发现 {len(images)} 个视觉元素，正在尝试提取文字...")
                for j, img in enumerate(images):
                    try:
                        # 从 pdfplumber 提取图片字节
                        bbox = (img["x0"], img["top"], img["x1"], img["bottom"])
                        cropped = page.within_bbox(bbox).to_image()
                        # 执行 OCR (尝试中文+英文)
                        ocr_text = pytesseract.image_to_string(cropped.original, lang='chi_sim+eng').strip()
                        if len(ocr_text) > 5:
                            vision_text_content += f"\n[图表/图片 {j+1} 内容]: {ocr_text}\n"
                    except Exception as e:
                        # 如果未安装 Tesseract 软件，这里会跳过
                        pass

            # 4. 语义组合
            filename = os.path.basename(file_path)
            vision_context = f"\n[🖼️ 视觉元素提示]: 本页包含 {len(images)} 个图表或图像。" if images else ""
            combined_content = f"--- Document: {filename} | Page: {i+1} ---\n{cleaned_text}\n\n{md_tables}{vision_context}{vision_text_content}"
            
            all_page_content.append({
                "page_content": combined_content,
                "metadata": {
                    "source": file_path, 
                    "page": i + 1, 
                    "filename": filename,
                    "has_images": len(images) > 0
                }
            })
            
    return all_page_content

def process_financial_reports_optimized():
    # ... (前面的路径设置保持不变)
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_script_dir)

    input_dir = os.path.join(project_root, "data")
    output_dir = os.path.join(project_root, "processed_chunks")

    if not os.path.exists(input_dir):
        print(f"❌ 错误：找不到数据文件夹！")
        return

    os.makedirs(output_dir, exist_ok=True)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
    )

    pdf_files = [f for f in os.listdir(input_dir) if f.endswith(".pdf")]

    for filename in pdf_files:
        file_path = os.path.join(input_dir, filename)
        output_filename = filename.replace(".pdf", "_chunks.json")
        output_path = os.path.join(output_dir, output_filename)

        if os.path.exists(output_path):
            print(f"⏭️ 跳过：{filename} 的切片已存在。")
            continue

        print(f"📄 正在执行语义清理与解析: {filename} ...")
        
        # 调用新版清理函数
        all_page_content = extract_with_layout_cleaning(file_path)

        final_chunks = []
        for page_data in all_page_content:
            splits = text_splitter.split_text(page_data["page_content"])
            for split in splits:
                final_chunks.append({
                    "content": split,
                    "metadata": page_data["metadata"]
                })

        output_data = []
        for j, chunk in enumerate(final_chunks):
            output_data.append({
                "chunk_id": j,
                "content": chunk["content"],
                "metadata": chunk["metadata"]
            })

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"  -> 💾 清理完成，保存了 {len(final_chunks)} 个高质量块。\n")

    print("🎉 [Step 1] 高保真解析优化完成！")


if __name__ == "__main__":
    process_financial_reports_optimized()