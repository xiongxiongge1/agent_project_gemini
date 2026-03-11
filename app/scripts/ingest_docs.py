import os
import asyncio
import sys

# 确保脚本能找到 app 目录
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.doc_loader import DocumentLoader
from app.services.knowledge_service import KnowledgeService

async def main():
    data_dir = "./data"
    loader = DocumentLoader()
    knowledge_service = KnowledgeService()

    if not os.path.exists(data_dir):
        print(f"错误: 找不到目录 {data_dir}")
        return

    # 扫描文件夹
    files = [f for f in os.listdir(data_dir) if f.endswith(('.pdf', '.docx'))]
    
    if not files:
        print("data 文件夹中没有找到 PDF 或 Word 文档。")
        return

    print(f"发现 {len(files)} 个文档，准备开始向量化...")

    for file_name in files:
        file_path = os.path.join(data_dir, file_name)
        print(f"正在处理: {file_name}...")

        try:
            # 1. 解析文档
            content = loader.load_document(file_path)
            
            # 2. 存入向量库
            # 元数据（Metadata）很重要，方便 Agent 溯源
            metadata = {
                "source": file_name,
                "type": os.path.splitext(file_name)[-1]
            }
            result = await knowledge_service.add_document(content, metadata)
            print(f"完成: {file_name} -> {result}")
            
        except Exception as e:
            print(f"处理 {file_name} 时出错: {str(e)}")

    print("\n所有文档向量化完毕！现在你的 Agent 可以查阅这些资料了。")

if __name__ == "__main__":
    asyncio.run(main())