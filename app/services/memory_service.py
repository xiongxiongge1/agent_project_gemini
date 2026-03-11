import json
import redis.asyncio as redis
from typing import List, Dict

class RedisMemoryService:
    def __init__(self):
        self.redis_client = redis.Redis(
            host = "192.168.101.68",
            port = 6379,
            password = 'redis',
            decode_responses = True # 关键：这会让 Redis 直接返回字符串而不是 bytes，省去我们手动解码的麻烦
        )
        self.ttl = 604800
        self.max_history_len = 20

    
    async def get_history(self, session_id: str) -> List[Dict]:
        """
        获取会话历史
        """
        key = f"chat_history:{session_id}"
        data = await self.redis_client.get(key)
        if data:
            return json.loads(data)
        return []
    
    async def add_interaction(self, session_id: str,user_query: str, agent_response: str):
        """
        将一轮新的对话（用户的提问 + Agent 的最终回答）存入 Redis
        """
        key = f"chat_history:{session_id}"
        # 1.先把老记忆拉出来
        history = await self.get_history(session_id)
        # 2. 追加新记忆
        history.append(
            {"role": "user", "content": user_query},
            
        )
        history.append(
            {"role": "assistant", "content": agent_response},
            
        )
        # 3. 核心机制：滑动窗口截断 (Sliding Window)
        # 如果历史记录太长，会把大模型的 Token 额度吃光，且很容易触发限流。这里只保留最近的 N 条。
        if len(history) > self.max_history_len:
            history = history[-self.max_history_len:]
        
        # 4. 存回 Redis 并刷新过期时间
        await self.redis_client.set(
            key,  # Redis key
            json.dumps(history, ensure_ascii=False),  # Redis value（序列化后的字符串）
            ex=self.ttl  # 过期时间（秒），对应 ttl；如果是毫秒用 px=self.ttl
        )

    async def clear_history(self, session_id: str):
        """
        清空会话历史
        """
        key = f"chat_history:{session_id}"
        await self.redis_client.delete(key)
