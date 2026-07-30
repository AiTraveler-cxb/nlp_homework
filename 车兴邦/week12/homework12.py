from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Iterable
from uuid import uuid4

from openai import OpenAI

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# 让本文件可以直接复用 src/tools.py
PROJECT_ROOT = Path(__file__).parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


# ============================================================
# 1. 多轮会话历史
# ============================================================

Message = dict[str, str]


@dataclass
class ConversationSession:
    """保存一个会话里的多轮最终问答。"""

    session_id: str = field(default_factory=lambda: uuid4().hex[:12])
    messages: list[Message] = field(default_factory=list)

    def add_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})

    def recent(self, max_turns: int = 6) -> list[Message]:
        """返回最近 max_turns 轮 user/assistant 历史。"""
        if max_turns <= 0:
            return []
        return self.messages[-max_turns * 2:]


def normalize_history(history: Iterable[Message] | None, max_turns: int = 6) -> list[Message]:
    """清洗外部传入历史，只保留 user/assistant 文本消息。"""
    if not history:
        return []
    cleaned: list[Message] = []
    for msg in history:
        role = msg.get("role")
        content = msg.get("content")
        if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
            cleaned.append({"role": role, "content": content.strip()})
    return cleaned[-max_turns * 2:]


# ============================================================
# 2. 手写 Prompt 解析版 ReAct
# ============================================================

MANUAL_CLIENT = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
MANUAL_MODEL = os.getenv("AGENT_MODEL", "qwen-max")

SYSTEM_PROMPT = """你是一个专业的A股金融分析助手，可以使用以下工具来回答问题：

工具列表：
1. rag_search(query) - 在年报中语义检索文本内容（战略/财务数据/风险因素等）
2. company_lookup(name) - 将公司名称转换为股票代码
3. calculator(expr) - 计算数学表达式（支持四则运算和math函数）
4. financial_indicator(symbol) - 获取实时财务指标（PE/PB/ROE等）
5. stock_price(symbol, start_date, end_date) - 获取历史股价，日期格式YYYYMMDD

你必须严格按照以下格式交替输出，每次只能调用一个工具：

Thought: 分析当前状态，决定下一步做什么
Action: 工具名称
Action Input: {"参数名": "参数值"}

收到工具结果后继续推理，直到可以给出最终答案：

Thought: 已有足够信息
Final Answer: 完整的回答（含数据来源）

规则：
- 必须先用 company_lookup 获取股票代码，再调用 financial_indicator 或 stock_price
- 数字计算必须用 calculator，不能心算
- Final Answer 必须引用具体数据来源（哪份年报哪一页，或AkShare实时数据）
- 如果没有合适工具能回答，直接输出 Final Answer 说明原因
"""

_THOUGHT_RE = re.compile(r"Thought:\s*(.+?)(?=\nAction:|\nFinal Answer:|$)", re.DOTALL)
_ACTION_RE = re.compile(r"Action:\s*(\w+)")
_ACTION_INPUT_RE = re.compile(r"Action Input:\s*(\{.+?\})", re.DOTALL)
_FINAL_RE = re.compile(r"Final Answer:\s*(.+)", re.DOTALL)


def parse_manual_step(text: str) -> dict:
    final = _FINAL_RE.search(text)
    if final:
        thought_m = _THOUGHT_RE.search(text)
        return {
            "type": "final",
            "thought": thought_m.group(1).strip() if thought_m else "",
            "answer": final.group(1).strip(),
        }

    thought_m = _THOUGHT_RE.search(text)
    action_m = _ACTION_RE.search(text)
    input_m = _ACTION_INPUT_RE.search(text)

    if not action_m:
        return {"type": "unparseable", "raw": text}

    try:
        action_input = json.loads(input_m.group(1)) if input_m else {}
    except json.JSONDecodeError:
        action_input = {}

    return {
        "type": "action",
        "thought": thought_m.group(1).strip() if thought_m else "",
        "action": action_m.group(1).strip(),
        "action_input": action_input,
    }


def run_manual(
    question: str,
    max_steps: int = 10,
    history: list[Message] | None = None,
    history_turns: int = 6,
) -> Generator[dict, None, None]:
    """手写版 ReAct，多轮能力由 history 参数提供。"""
    from tools import TOOLS_MAP

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *normalize_history(history, history_turns),
        {"role": "user", "content": question},
    ]

    for step in range(1, max_steps + 1):
        response = MANUAL_CLIENT.chat.completions.create(
            model=MANUAL_MODEL,
            messages=messages,
            temperature=0,
            stop=["Observation:"],
        )
        llm_output = response.choices[0].message.content.strip()
        parsed = parse_manual_step(llm_output)

        if parsed["type"] == "final":
            yield {"step": step, "type": "final", "thought": parsed["thought"], "answer": parsed["answer"]}
            return

        if parsed["type"] == "unparseable":
            yield {"step": step, "type": "error", "observation": f"格式解析失败：{llm_output[:200]}"}
            return

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

        yield {
            "step": step,
            "type": "action",
            "thought": parsed["thought"],
            "action": tool_name,
            "action_input": tool_args,
            "observation": str(observation),
        }

        # 本轮 scratchpad：追加模型动作和工具 observation，继续推理。
        messages.append({"role": "assistant", "content": llm_output})
        messages.append({"role": "user", "content": f"Observation: {observation}\n"})

    yield {"step": max_steps + 1, "type": "max_steps", "answer": f"已达最大步数 {max_steps}，未能得出最终答案"}


# ============================================================
# 3. Function Calling 版 ReAct
# ============================================================

FC_CLIENT = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)
FC_MODEL = os.getenv("AGENT_MODEL", "deepseek-v4-flash")

FC_SYSTEM_PROMPT = """你是一个专业的A股金融分析助手。
规则：
- 调用 financial_indicator 或 stock_price 之前，必须先用 company_lookup 获取股票代码
- 数字计算必须使用 calculator 工具，不能心算
- Final Answer 必须引用具体数据来源
- 如果没有合适工具能回答，直接说明原因
"""


def run_fc(
    question: str,
    max_steps: int = 10,
    history: list[Message] | None = None,
    history_turns: int = 6,
) -> Generator[dict, None, None]:
    """Function Calling 版 ReAct，多轮能力由 history 参数提供。"""
    from tools import TOOLS_MAP, TOOLS_SCHEMA

    messages = [
        {"role": "system", "content": FC_SYSTEM_PROMPT},
        *normalize_history(history, history_turns),
        {"role": "user", "content": question},
    ]

    for step in range(1, max_steps + 1):
        response = FC_CLIENT.chat.completions.create(
            model=FC_MODEL,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
            temperature=0,
        )
        msg = response.choices[0].message
        reason = response.choices[0].finish_reason

        if reason == "stop" or not msg.tool_calls:
            yield {"step": step, "type": "final", "thought": "", "answer": msg.content or "（模型返回空内容）"}
            return

        messages.append(msg)

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

            yield {
                "step": step,
                "type": "action",
                "thought": "",
                "action": tool_name,
                "action_input": tool_args,
                "observation": str(observation),
            }

            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": str(observation)})

    yield {"step": max_steps + 1, "type": "max_steps", "answer": f"已达最大步数 {max_steps}，未能得出最终答案"}


# ============================================================
# 4. CLI 展示
# ============================================================

DEFAULT_QUESTION = "贵州茅台和五粮液2023年的毛利率哪家更高？差多少个百分点？"


def get_runner(mode: str):
    return run_manual if mode == "manual" else run_fc


def print_step(step_data: dict, mode: str) -> str:
    """打印一步，返回 final answer 或空字符串。"""
    stype = step_data["type"]
    if stype == "action":
        print(f"\n[Step {step_data['step']}]")
        if mode == "manual" and step_data.get("thought"):
            print(f"Thought: {step_data['thought']}")
        elif mode == "fc":
            print("Thought: Function Calling 版内部推理，不可见")
        print(f"Action: {step_data['action']}")
        print(f"Action Input: {json.dumps(step_data['action_input'], ensure_ascii=False)}")
        print(f"Observation: {step_data['observation'][:500]}")
        return ""
    if stype == "final":
        print(f"\n✅ Final Answer:\n{step_data['answer']}")
        return step_data["answer"]
    if stype in {"error", "max_steps"}:
        answer = step_data.get("answer", step_data.get("observation", ""))
        print(f"\n⚠️ {answer}")
        return answer
    return ""


def run_once(mode: str, question: str, max_steps: int):
    print(f"问题：{question}")
    print(f"模式：{mode}")
    start = time.time()
    runner = get_runner(mode)
    for step in runner(question, max_steps=max_steps):
        print_step(step, mode)
    print(f"耗时：{time.time() - start:.1f}s")


def chat_loop(mode: str, max_steps: int, history_turns: int):
    """多轮对话入口。"""
    session = ConversationSession()
    runner = get_runner(mode)
    print(f"多轮对话已启动，mode={mode}, session={session.session_id}")
    print("命令：/new 新会话，/history 查看历史，/exit 退出")

    while True:
        question = input("\n你：").strip()
        if not question:
            continue
        if question == "/exit":
            print("再见！")
            break
        if question == "/new":
            session = ConversationSession()
            print(f"已开始新会话：{session.session_id}")
            continue
        if question == "/history":
            print(json.dumps(session.messages, ensure_ascii=False, indent=2))
            continue

        print(f"\n[加载历史] 最近 {len(session.recent(history_turns)) // 2} 轮")
        final_answer = ""
        for step in runner(question, max_steps=max_steps, history=session.recent(history_turns), history_turns=history_turns):
            maybe_answer = print_step(step, mode)
            if maybe_answer:
                final_answer = maybe_answer

        if final_answer:
            session.add_user(question)
            session.add_assistant(final_answer)
            print(f"\n[已写入会话历史] 当前共 {len(session.messages) // 2} 轮")


def main():
    parser = argparse.ArgumentParser(description="作业12：多轮对话 ReAct Agent")
    parser.add_argument("--mode", choices=["manual", "fc"], default="manual")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--history_turns", type=int, default=6)
    parser.add_argument("--chat", action="store_true", help="进入多轮对话模式")
    args = parser.parse_args()

    if args.chat:
        chat_loop(args.mode, args.max_steps, args.history_turns)
    else:
        run_once(args.mode, args.question, args.max_steps)


if __name__ == "__main__":
    main()
