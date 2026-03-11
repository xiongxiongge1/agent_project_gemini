import json
import re
from typing import List, Dict
from openai import AsyncOpenAI
from app.core.config import settings
from app.tools.base_tools import AVAILABLE_TOOLS
from app.services.memory_service import RedisMemoryService
import tiktoken 

class ReActAgent:
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=settings.DASHSCOPE_API_KEY,
            base_url=settings.DASHSCOPE_API_URL
        )
        self.memory :  Dict[str, List[Dict]] = {}
        self.max_history_len = 10  # 超过 10 条开始摘要
        # 摘要存储：{session_id: "之前谈话的简要背景..."}
        self.summarizes: Dict[str, List[dict]] = {}
        self.summary_threshold = 6 # 每次将最老的 6 条摘要成一段话
        self.memory_service = RedisMemoryService()
        self.base_system_prompt  = """你是一个具备思考能力的 AI 助手。你可以使用以下工具：
            1. get_weather: 参数 {"city": "城市名称"}
            2. get_stock_price: 参数 {"symbol": "股票代码"}
            3. search_knowledge: 搜索合同相关知识，参数 {"query": "搜索内容"}
            你必须严格遵守以下格式回复：
            Thought: 思考你应该做什么
            Action: 工具名称
            Action Input: 工具参数 (JSON 格式)
            Observation: 工具返回的结果
            ... (重复上述步骤直到获得答案)
            Final Answer: 最终给用户的回答"""


    async def _get_summary_model_response(self, content_to_summarize: str) -> str:
        """调用低成本模型生成摘要"""
        # 这里可以使用更便宜的模型，如 qwen-turbo
        prompt = f"请简要总结以下对话内容的要点，用于后续对话背景。要求简洁有力，不超过100字：\n\n{content_to_summarize}"
        response = await self.client.chat.completions.create(
            model="qwen-turbo",
            messages=[
                {"role": "user", "content": prompt},
            ],
            stream=False,
            extra_body={"enable_thinking": False}
        )
        return response.choices[0].message.content
    
    async def _get_manage_memory(self, session_id: str):
        """核心记忆管理逻辑：摘要 + 修剪"""
        history = await self.memory_service.get_history(session_id)
        # 如果历史消息（除去 System）超过阈值
        real_history = [msg for msg in history if msg["role"] != "system"]
        if len(real_history) > self.max_history_len:
            print(f"[Memory] Session {session_id} 触发摘要机制...")

            # 1. 提取最老的N条对话（跳过第0条 System Prompt）
            to_summarize = real_history[:self.summary_threshold]
            # 2.剩余对话
            remaining = real_history[self.summary_threshold:]

            # 3.格式化待摘要内容
            text_to_summarize = ""
            for msg in to_summarize:
                role = "用户" if msg["role"] == "user" else "助手"
                text_to_summarize += f"{role}: {msg['content']}\n"

            # 4.生成新摘要并更新
            prev_summary = self.summarizes.get(session_id, "无")
            new_chunk_summary = await self._get_summary_model_response(f"已知背景: {prev_summary}\n新对话内容: {text_to_summarize}")
            self.summarizes[session_id] = new_chunk_summary

            # 5. 更新内容： System Prompt + 新的历史记录
            self.memory[session_id]  =[history[0]] + remaining
            # self.memory_sevice.clear_history(session_id)
            # TODO 如何把简化后的历史记录存储到redis中
            # await self.memory_service.add_interaction(session_id, text_to_summarize, new_chunk_summary)
            print(f"[Memory] 新摘要已生成: {new_chunk_summary}")

    def _init_session(self, session_id: str):
        """初始化会话记忆（确保System Prompt唯一）"""
        if session_id not in self.memory:
            self.memory[session_id] = [{"role": "system", "content": self.base_system_prompt}]
        if session_id not in self.summarizes:
            self.summarizes[session_id] = "无"

    def _get_history(self, session_id: str):
        if session_id not in self.memory:
            self.memory[session_id] = [{"role": "system","content": self.base_system_prompt}]
        return self.memory[session_id]
    def _get_clean_history(self, session_id: str) -> List[dict]:
        """获取经过窗口裁剪的历史记录"""
        if session_id not in self.memory:
            self.memory[session_id] = [{"role": "system","content": self.base_system_prompt}]
        history = self.memory[session_id]

        if len(history) > self.max_history_len:
            history = [history[0]] + history[-(self.max_history_len ):]
            self.memory[session_id] = history
        return history



    def _num_tokens_from_messages(self, messages: List[Dict]) -> int:
        """
        精确计算消息列表消耗的 Token 数。
        注意：使用 cl100k_base 编码（适用于 GPT-4, Qwen, Llama3 等主流模型）。
        """
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            # 如果联网获取失败，回退到默认编码
            encoding = tiktoken.get_encoding("gpt2")
            
        num_tokens = 0
        for message in messages:
            # 每条消息的基础开销：<|start|>role<|message|>...<|end|>
            num_tokens += 4  
            for key, value in message.items():
                num_tokens += len(encoding.encode(str(value)))
                if key == "name":  # 如果有名字字段，额外增加开销
                    num_tokens += -1
        num_tokens += 2  # 所有的回答都以 <|start|>assistant 开头
        return num_tokens

    async def run(self, query: str, session_id: str):
        # 初始化会话
        # self._init_session(session_id)
        chat_history = await self.memory_service.get_history(session_id)
        # 获取当前摘要
        current_summary = self.summarizes.get(session_id, "暂无背景信息")

        # 构建动态 System Prompt
        dynamic_system_prompt = f"""你是一个具备工具调用能力的助手。
        [已知对话背景]: {current_summary},
        {self.base_system_prompt}
        """
        # 构造本次请求的上下文
        working_messages = []
        working_messages.append({"role": "system", "content": dynamic_system_prompt})
        for msg in chat_history:
            working_messages.append(msg)
        working_messages.append({"role": "user", "content": query})
        # 1. 获取裁剪后的长时记忆
        # messages = self._get_clean_history(session_id)
        # 2. 构造当前 ReAct 循环的临时上下文 (Working Memory)

        final_answer = ""

        print(f"\n[Session: {session_id}] New Query: {query}")
        print("-" * 50)

        for i in range(5):
            print(f"\n[Round {i+1}]: Agent is thinking...")
            
            # 注意：这里我们拿到了模型一整轮的输出
            response = await self.client.chat.completions.create(
                model=settings.MODEL_NAME,
                messages=working_messages,
                extra_body={"enable_thinking": False},
                stop=["Observation:", "Observation"] 
            )
            
            content = response.choices[0].message.content
            
            # --- 控制台实时打印模型输出 ---
            print(f"\033[32m{content}\033[0m") # 绿色显示模型思考过程
            
            # 发送到前端接口
            yield f"data: {json.dumps({'type': 'thought', 'content': content}, ensure_ascii=False)}\n\n"
            
            if "Final Answer:" in content:
                # messages.append({"role": "assistant", "content": content})
                final_answer = content.split("Final Answer:")[-1].strip()
                print("-" * 50)
                print("[Agent]: Task Completed.\n")
                break
            
            # 解析逻辑
            action_match = re.search(r"Action:\s*(.*)", content)
            action_input_match = re.search(r"Action Input:\s*(.*)", content)
            
            if action_match and action_input_match:
                tool_name = action_match.group(1).strip()
                tool_input_str = action_input_match.group(1).strip()
                
                print(f"[System]: Calling Tool -> {tool_name} with {tool_input_str}")
                
                try:
                    tool_input = json.loads(tool_input_str)
                    tool_func = AVAILABLE_TOOLS.get(tool_name)
                    if tool_func:
                        # 核心：判断是否是异步函数并正确调用
                        import inspect
                        if inspect.iscoroutinefunction(tool_func):
                            observation = await tool_func(**tool_input)
                        else:
                            observation = tool_func(**tool_input)
                    else:
                        observation = f"Error: 工具 {tool_name} 不存在"
                except Exception as e:
                    observation = f"Error: {str(e)}"
                
                print(f"\033[34m[Observation]: {observation}\033[0m") # 蓝色显示工具返回
                
                # 发送观察结果到前端
                yield f"data: {json.dumps({'type': 'observation', 'content': str(observation)}, ensure_ascii=False)}\n\n"
                
                working_messages.append({"role": "assistant", "content": content})
                working_messages.append({"role": "user", "content": f"Observation: {observation}"})
            else:
                # 如果模型没有按格式回复，强制结束防止死循环
                print("[System]: Format Error, stopping.")
                break
        # 3. 任务结束：将“干净”的结果存入长时记忆
        # 这样下次请求时，history 里只有 User 的问题和 Assistant 的最终回答
        # self.memory[session_id].append({"role": "user", "content": query})
        await self.memory_service.add_interaction(session_id, query, final_answer)
        # self.memory[session_id].append({"role": "assistant", "content": f"Final Answer: {final_answer}"})
        # 检查是否需要压缩记忆到摘要
        # current_history = await self.memory_service.get_history(session_id)
        # current_tokens = self._num_tokens_from_messages(current_history)
        # print(f"[Monitor] 当前会话 Token 数: {current_tokens}")
        # if current_tokens > 2000: # 假设 2000 是你的预警线
        #     await self._get_manage_memory(session_id)
        # print(f"[System]: Memory updated. Current count: {len(current_history)}")