import uuid
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.services.vector_service import VectorService
from app.services.rerank_service import RerankService
import jieba # 新增中文分词
from rank_bm25 import BM25Okapi # 新增关键词检索库

class KnowledgeService:
    def __init__(self):
        # 1. 初始化本地持久化数据库（数据会存在当前目录下的 ./chroma_db 文件夹）
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        # 2. 创建或获取一个名为 "science_papers" 的集合
        self.collection = self.chroma_client.get_or_create_collection(name="science_papers")
        self.vector_service = VectorService()
        self.rerank_service = RerankService() # 2. 初始化 Rerank 服务
        self.bm25 = None
        self.all_documents = []
        self._init_bm25_index()

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
        self._init_bm25_index()
        return f"成功存储 {len(chunks)} 个知识块"
    def _init_bm25_index(self):
        """从 ChromaDB 加载所有文档，构建 BM25 关键词索引"""
        all_data = self.collection.get()
        self.all_documents = all_data.get('documents', [])
        
        if self.all_documents:
            print(f"[Knowledge] 正在为 {len(self.all_documents)} 个知识块构建 BM25 关键词索引...")
            # 对所有文档进行分词
            tokenized_corpus = [list(jieba.cut(doc)) for doc in self.all_documents]
            self.bm25 = BM25Okapi(tokenized_corpus)
            print("[Knowledge] BM25 索引构建完成！")
        else:
            print("[Knowledge] 知识库为空，暂不构建 BM25 索引。")

    # 注意：如果你写了一个向 ChromaDB 添加新文档的方法 (add_document)，
    # 记得在那个方法的最后加上 `self._init_bm25_index()` 来刷新关键词索引。

    # ================= 新增：RRF 融合算法 =================
    def _rrf_fusion(self, vector_docs: list, bm25_docs: list, k: int = 60) -> list:
        """
        Reciprocal Rank Fusion (倒数排名融合)
        公式: Score = 1 / (k + rank)
        """
        rrf_scores = {}
        
        # 给向量检索的结果打分
        for rank, doc in enumerate(vector_docs):
            if doc not in rrf_scores:
                rrf_scores[doc] = 0.0
            rrf_scores[doc] += 1.0 / (k + rank + 1)
            
        # 给 BM25 检索的结果打分
        for rank, doc in enumerate(bm25_docs):
            if doc not in rrf_scores:
                rrf_scores[doc] = 0.0
            rrf_scores[doc] += 1.0 / (k + rank + 1)
            
        # 按总得分从高到低排序，提取合并后的文档
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in sorted_docs]
    
    async def search(self, query: str, top_k: int = 3):
        """三阶段检索：向量召回 + BM25召回 -> RRF 融合 -> 大模型 Rerank 精排"""
        fetch_k = 15 
        
        # ---------------- 1. 向量检索 (语义) ----------------
        print(f"\n\033[96m[Search] 路召回 1/2：正在进行向量检索...\033[0m")
        query_vector = await self.vector_service.get_embeddings([query])
        vector_results = self.collection.query(
            query_embeddings=query_vector,
            n_results=fetch_k
        )
        vector_docs = vector_results['documents'][0] if vector_results['documents'] else []

        # ---------------- 2. BM25 检索 (关键词精确匹配) ----------------
        print(f"\033[96m[Search] 路召回 2/2：正在进行 BM25 关键词检索...\033[0m")
        bm25_docs = []
        if self.bm25 and self.all_documents:
            tokenized_query = list(jieba.cut(query))
            # 获取 top_k 结果
            bm25_docs = self.bm25.get_top_n(tokenized_query, self.all_documents, n=fetch_k)

        # ---------------- 3. RRF 融合 (去重并按权重合并) ----------------
        fused_docs = self._rrf_fusion(vector_docs, bm25_docs)
        # 截取融合后的前 20 条交给 Rerank，防止 Rerank 压力过大
        fused_docs = fused_docs[:20] 
        print(f"\033[94m[Search] RRF 融合完毕，共产生 {len(fused_docs)} 条独特候选文本。\033[0m")

        if not fused_docs:
            print("[Search] 未检索到任何相关内容。")
            return []

        # ---------------- 4. 大模型 Rerank 精排 ----------------
        print(f"\033[93m[Search] 终极阶段：启动 Rerank 重新排序...\033[0m")
        final_docs = await self.rerank_service.rerank(
            query=query,
            documents=fused_docs,
            top_n=top_k
        )
        
        print(f"\n\033[92m=== 混合检索最终采纳的前 {len(final_docs)} 条 ===\033[0m")
        for i, doc in enumerate(final_docs):
            preview = doc.replace('\n', ' ')[:60]
            print(f"[{i+1}] {preview}...")
        print("\033[92m========================================\033[0m\n")
        
        return final_docs