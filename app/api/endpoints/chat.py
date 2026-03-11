import asyncio
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.schemas.chat import ChatRequest
from app.services.agent_service import ReActAgent
router = APIRouter()
agent = ReActAgent()
async def fake_agent_stream(query : str):
    """
    模拟一个 Agent 的思考逻辑
    在实际开发中，这里会调用大模型的chat completion接口
    """

    responses = [
        f"Thought: 正在处理关于 '{query}' 的请求...\n",
        "Action: 调用搜索工具...\n",
        "Observation: 获取到相关数据：'Agent 工程化架构是基础'。\n",
        "Final Answer: 搭建 FastAPI 骨架是成为 Agent 工程师的第一步。"
    ]
    for chunk in responses:
        yield f"data: {chunk} \n\n"
        await asyncio.sleep(0.6) #模拟延迟

@router.post("/chat")
async def chat_endpoint(request: ChatRequest):
    return StreamingResponse(agent.run(request.query, request.session_id), media_type="text/event-stream")

