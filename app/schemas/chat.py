from pydantic import BaseModel, Field
from typing import Optional

class ChatRequest(BaseModel):
    query:str = Field(..., description="用户发送的消息")
    session_id: str = Field( default = "default_user", description="会话ID")
    stream: bool = Field(default=True, description="是否流式返回结果")


class ChatResponse(BaseModel):
    messsage : str
    status: str = 'success'