class ReActAgent:
    def __init__(self, system_prompt=FC_SYSTEM_PROMPT, max_steps=10):
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.messages = [{"role": "system", "content": system_prompt}]
        self.tools = TOOLS_SCHEMA  # 从tools导入
        self.tool_map = TOOLS_MAP

    def chat(self, user_input: str) -> str:
        # 添加用户消息
        self.messages.append({"role": "user", "content": user_input})
        # 执行循环
        for step in range(1, self.max_steps + 1):
            response = client.chat.completions.create(
                model=MODEL,
                messages=self.messages,
                tools=self.tools,
                tool_choice="auto",
                temperature=0,
            )
            msg = response.choices[0].message
            reason = response.choices[0].finish_reason

            if reason == "stop" or not msg.tool_calls:
                # 最终答案
                final_answer = msg.content or "（模型返回空内容）"
                self.messages.append({"role": "assistant", "content": final_answer})
                return final_answer

            # 有工具调用
            self.messages.append(msg)  # assistant消息包含tool_calls
            for tool_call in msg.tool_calls:
                tool_name = tool_call.function.name
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except:
                    tool_args = {}
                tool_fn = self.tool_map.get(tool_name)
                if tool_fn is None:
                    observation = f"未知工具 '{tool_name}'"
                else:
                    try:
                        observation = tool_fn(**tool_args)
                    except Exception as e:
                        observation = f"工具执行错误: {e}"
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(observation),
                })
        # 达到最大步数
        error_msg = f"已达最大步数 {self.max_steps}，未能得出最终答案"
        self.messages.append({"role": "assistant", "content": error_msg})
        return error_msg
