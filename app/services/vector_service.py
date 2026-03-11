import json
from openai import AsyncOpenAI
from app.core.config import settings
import asyncio

class VectorService:
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=settings.DASHSCOPE_API_KEY,
            base_url=settings.DASHSCOPE_API_URL
            )
        self.model = "text-embedding-v3"
    
    async def get_embeddings(self, text_list: list[str]):
        """将文本列表转换为向量列表"""
        batch_size = 10
        all_embeddings = []
        for i in range(0, len(text_list), batch_size):
            batch = text_list[i:i+batch_size]
            print(f"[Embedding] 正在处理第 {i//batch_size + 1} 批数据...")

            response = await self.client.embeddings.create(
                model=self.model,
                input=batch
            )
            batch_embeddings = [data.embedding for data in response.data]
            all_embeddings.extend(batch_embeddings)
            await asyncio.sleep(0.1)
        return all_embeddings
# 简单测试逻辑
if __name__ == "__main__":
    service = VectorService()
    vectors = asyncio.run(service.get_embeddings(["分子对接工具", "GNN图神经网络"]))
    print(len(vectors[0])) # 通常是 1024 或 1536 维 