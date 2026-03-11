import httpx
import json
from typing import List
from app.core.config import settings

class RerankService:
    def __init__(self):
        self.api_key = settings.DASHSCOPE_API_KEY
        # 阿里云 DashScope 的兼容 Rerank API
        self.url = "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"
        self.model = "qwen3-rerank"

    async def rerank(self, query: str, documents: List[str], top_n: int = 3) -> List[str]:
        """
        调用阿里云 DashScope Rerank API 进行深度重排序
        """
        if not documents:
            return []

        # 防止请求的 top_n 超过文档总数
        top_n = min(top_n, len(documents))

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # 根据你提供的 curl 构建的请求体
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "instruct": "Given a web search query, retrieve relevant passages that answer the query." # qwen3-rerank 的建议指令
        }

        print(f"\033[93m[Rerank] 正在对 {len(documents)} 条粗排文档进行精排...\033[0m")

        try:
            # 发起异步 POST 请求，Rerank 比较耗时，建议 timeout 设置长一点
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.url, 
                    headers=headers, 
                    json=payload, 
                    timeout=15.0
                )
                
                # 如果请求失败，抛出异常或使用备用方案
                if response.status_code != 200:
                    print(f"[Rerank] API 错误 ({response.status_code}): {response.text}")
                    # 容灾降级：如果 Rerank 失败，直接返回原向量检索的前 top_n 条
                    return documents[:top_n]

                resp_json = response.json()
                
                # 1. 提取 results 列表 (兼容阿里云的格式)
                results = resp_json.get("results") or resp_json.get("output", {}).get("results", [])
                
                if not results:
                    print(f"\033[91m[Rerank 警告] API 返回格式异常。完整响应: {resp_json}\033[0m")
                    return documents[:top_n]

                # 2. 根据返回的 index，去原文档列表里找对应的文本
                reranked_docs = []
                for item in results:
                    idx = item.get("index")
                    score = item.get("relevance_score", 0.0)
                    
                    # 只要 index 存在且没越界，就把文本拿出来
                    if idx is not None and idx < len(documents):
                        doc_text = documents[idx]
                        print(f"\033[92m[Rerank Score] {score:.4f} -> {doc_text[:40]}...\033[0m")
                        reranked_docs.append(doc_text)
                
                # 如果因为某些原因没解析到，走降级
                if not reranked_docs:
                    return documents[:top_n]

                return reranked_docs

        except Exception as e:
            print(f"\033[91m[Rerank 崩溃] {str(e)}\033[0m")
            return documents[:top_n]