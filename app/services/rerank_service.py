import httpx
import json
from typing import List
from app.core.config import settings

class RerankService:
    def __init__(self):
        self.api_key = settings.DASHSCOPE_API_KEY
        self.url = "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"
        self.model = "qwen3-rerank" 

    async def rerank(self, query: str, documents: List[str], top_n: int = 3) -> List[str]:
        if not documents:
            return []

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "return_documents": True 
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(self.url, headers=headers, json=payload, timeout=10.0)
                
                if response.status_code != 200:
                    print(f"Rerank API 错误: {response.text}")
                    return documents[:top_n]

                resp_json = response.json()
                
                # --- 核心解析逻辑：匹配你提供的 JSON 结构 ---
                # 路径：output -> results -> [index/document/relevance_score]
                results = resp_json.get("output", {}).get("results", [])
                
                reranked_docs = []
                for item in results:
                    # 优先获取 API 直接返回的文本内容
                    doc_text = item.get("document", {}).get("text")
                    if not doc_text:
                        # 兜底方案：如果 API 没给 text，通过原始索引去找
                        index = item.get("index")
                        doc_text = documents[index]
                    
                    # 打印分数（可选，方便调试）
                    score = item.get("relevance_score")
                    print(f"\033[92m[Rerank Score]: {score:.4f} -> {doc_text[:30]}...\033[0m")
                    
                    reranked_docs.append(doc_text)

                return reranked_docs

        except Exception as e:
            print(f"Rerank 异常: {str(e)}")
            return documents[:top_n]