def run(question: str, max_steps: int = 10, history: list = None) -> Generator[dict, None, None]:
    from tools import TOOLS_MAP, TOOLS_SCHEMA

    if history is None:
        messages = [
            {"role": "system", "content": FC_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
    else:
        messages = history
        messages.append({"role": "user", "content": question})

    for step in range(1, max_steps + 1):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
            temperature=0,
        )
        msg = response.choices[0].message
        reason = response.choices[0].finish_reason

        if reason == "stop" or not msg.tool_calls:
            # 直接回答
            messages.append({"role": "assistant", "content": msg.content or ""})
            yield {
                "step": step,
                "type": "final",
                "thought": "",
                "answer": msg.content or "（模型返回空内容）",
            }
            return

        # 模型请求调用工具
        messages.append(msg)  # 保留 assistant 的 tool_calls 信息

        for tool_call in msg.tool_calls:
            tool_name = tool_call.function.name
            try:
                tool_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                tool_args = {}

            tool_fn = TOOLS_MAP.get(tool_name)
            if tool_fn is None:
                observation = f"未知工具 '{tool_name}'"
            else:
                try:
                    observation = tool_fn(**tool_args)
                except TypeError as e:
                    observation = f"工具参数错误: {e}"

            step_result = {
                "step": step,
                "type": "action",
                "thought": "",
                "action": tool_name,
                "action_input": tool_args,
                "observation": str(observation),
            }
            yield step_result

            # 将工具返回结果加入历史
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(observation),
            })

    # 达到最大步数
    messages.append({"role": "assistant", "content": f"已达最大步数 {max_steps}，未能得出最终答案"})
    yield {
        "step": max_steps + 1,
        "type": "max_steps",
        "answer": f"已达最大步数 {max_steps}，未能得出最终答案",
    }
