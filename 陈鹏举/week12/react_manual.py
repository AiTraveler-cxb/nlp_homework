def run(question: str, max_steps: int = 10, history: list = None) -> Generator[dict, None, None]:
    """
    执行 ReAct 循环
    :param question: 当前用户问题
    :param max_steps: 最大推理步数
    :param history: 已有的消息列表（包含 system 及之前的对话），若为 None 则新建
    """
    from tools import TOOLS_MAP

    if history is None:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": question},
        ]
    else:
        messages = history
        messages.append({"role": "user", "content": question})

    # 后续循环中，所有 append 都直接作用在 messages 上
    for step in range(1, max_steps + 1):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0,
            stop=["Observation:"],
        )
        llm_output = response.choices[0].message.content.strip()
        parsed = _parse_step(llm_output)

        if parsed["type"] == "final":
            # 将 assistant 的最终回答加入历史
            messages.append({"role": "assistant", "content": llm_output})
            yield {
                "step": step,
                "type": "final",
                "thought": parsed["thought"],
                "answer": parsed["answer"],
            }
            return

        if parsed["type"] == "unparseable":
            messages.append({"role": "assistant", "content": llm_output})
            yield {
                "step": step,
                "type": "error",
                "observation": f"格式解析失败，原始输出：{llm_output[:200]}",
            }
            return

        # 执行工具
        tool_name = parsed["action"]
        tool_args = parsed["action_input"]
        tool_fn = TOOLS_MAP.get(tool_name)
        if tool_fn is None:
            observation = f"未知工具 '{tool_name}'，可用工具：{list(TOOLS_MAP.keys())}"
        else:
            try:
                observation = tool_fn(**tool_args)
            except TypeError as e:
                observation = f"工具参数错误: {e}"

        # 将 assistant 输出和工具观察结果加入历史
        messages.append({"role": "assistant", "content": llm_output})
        messages.append({"role": "user", "content": f"Observation: {observation}"})

        step_result = {
            "step": step,
            "type": "action",
            "thought": parsed["thought"],
            "action": tool_name,
            "action_input": tool_args,
            "observation": str(observation),
        }
        yield step_result

    # 达到最大步数
    messages.append({"role": "assistant", "content": f"已达最大步数 {max_steps}，未能得出最终答案"})
    yield {
        "step": max_steps + 1,
        "type": "max_steps",
        "answer": f"已达最大步数 {max_steps}，未能得出最终答案",
    }
