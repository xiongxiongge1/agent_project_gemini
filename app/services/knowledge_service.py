import uuid
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.services.vector_service import VectorService
from app.services.rerank_service import RerankService

class KnowledgeService:
    def __init__(self):
        # 1. 初始化本地持久化数据库（数据会存在当前目录下的 ./chroma_db 文件夹）
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        # 2. 创建或获取一个名为 "science_papers" 的集合
        self.collection = self.chroma_client.get_or_create_collection(name="science_papers")
        self.vector_service = VectorService()
        self.rerank_service = RerankService() # 2. 初始化 Rerank 服务

# 3. 定义分块器：每块 500 字，重叠 50 字（防止语义在切分处断开）
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len)
    

    async def add_document(self, text: str, metadata: dict):
        """将长文档切块并存入向量库"""
        # 1. 原始切分
        raw_chunks = self.text_splitter.split_text(text)
        
        # 2. 【核心修复】过滤空字符串和纯空白符，并去除首尾空格
        # 这一步能确保 chunks 的数量和后面生成的 embeddings 数量严格一致
        chunks = [c.strip() for c in raw_chunks if c.strip()]
        
        if not chunks:
            return "文档解析后无有效内容，跳过存储"

        # 3. 此时再获取向量
        embeddings = await self.vector_service.get_embeddings(chunks)
        
        # 4. 确保 ID 和元数据与过滤后的 chunks 数量匹配
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [metadata for _ in chunks]
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas
        )
        return f"成功存储 {len(chunks)} 个知识块"
    
    async def search(self, query: str, top_k: int = 3):
        """两阶段检索：向量召回 (粗排) + 大模型重排序 (精排)"""
        
        # 阶段一：粗排 (扩大搜索范围，比如原本要 3 条，现在捞 15 条)
        fetch_k = 15 
        print(f"[Search] 阶段一：正在通过向量检索获取前 {fetch_k} 条候选内容...")
        query_vector = await self.vector_service.get_embeddings([query])

        results = self.collection.query(
            query_embeddings=query_vector,
            n_results=fetch_k
        )
        
        retrieved_docs = results['documents'][0] if results['documents'] else []
        print(f"\n\033[90m=== 向量检索 (粗排) 前 3 条预览 ===\033[0m")
        for i, doc in enumerate(retrieved_docs[:3]):
            # 替换换行符方便单行查看，只截取前 80 个字符
            preview = doc.replace('\n', ' ')[:80]
            print(f"[{i+1}] {preview}...")
        print("\033[90m======================================\033[0m\n")
        if not retrieved_docs:
            return []

        # 阶段二：精排 (使用 Rerank 模型挑选最相关的 top_k 条)
        print(f"[Search] 阶段二：启动 Rerank 重新排序...")
        final_docs = await self.rerank_service.rerank(
            query=query,
            documents=retrieved_docs,
            top_n=top_k
        )
        
        return final_docs