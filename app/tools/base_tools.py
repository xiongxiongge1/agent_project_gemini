import json
from app.services.knowledge_service import KnowledgeService


knowledge_service = KnowledgeService()

async def search_knowledge(query: str):
    """
    当你不确定某个专业领域（如分子对接、GNN、特定论文内容）的知识时，使用此工具。
    参数 query: 搜索关键词或问题。
    """
    results = await knowledge_service.search(query)
    return "\n".join(results) if results else "未在知识库中找到相关信息。"

def get_weather(city: str):
    """获取天气信息"""

    data = {"北京": "晴，25度", "上海": "小雨，20度", "广州": "多云，28度"}
    return data.get(city, "没有该城市的天气信息")

def get_stock_price(symbol: str):

    prices = {"AAPL": "180 USD", "NVDA": "900 USD", "BABA": "75 USD"}
    return prices.get(symbol, "没有该股票的报价")

# 工具映射表，方便 Agent 根据名称找到函数 
AVAILABLE_TOOLS = {
    "get_weather": get_weather,
    "get_stock_price": get_stock_price,
    "search_knowledge": search_knowledge # 注册新工具
}