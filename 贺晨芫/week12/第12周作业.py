"""
手写 Prompt 解析版 ReAct Agent
"""

import os
import re
import json
import time
import logging
import argparse
from typing import Generator

from openai import OpenAI

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ── LLM 客户端 ────────────────────────────────────────────────────────────────
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
MODEL = os.getenv("AGENT_MODEL", "qwen-max")
# client = OpenAI(
#     api_key=os.getenv("DEEPSEEK_API_KEY"),
#     base_url="https://api.deepseek.com",
# )
# MODEL = os.getenv("AGENT_MODEL", "deepseek-v4-flash")


# ── System Prompt ─────────────────────────────────────────────────────────────
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

# ── 格式解析 ──────────────────────────────────────────────────────────────────
_THOUGHT_RE      = re.compile(r"Thought:\s*(.+?)(?=\nAction:|\nFinal Answer:|$)", re.DOTALL)
_ACTION_RE       = re.compile(r"Action:\s*(\w+)")
_ACTION_INPUT_RE = re.compile(r"Action Input:\s*(\{.+?\})", re.DOTALL)
_FINAL_RE        = re.compile(r"Final Answer:\s*(.+)", re.DOTALL)


def _parse_step(text: str) -> dict:
    """从 LLM 输出中解析一步的结构化内容"""
    final = _FINAL_RE.search(text)
    if final:
        thought_m = _THOUGHT_RE.search(text)
        return {
            "type":    "final",
            "thought": thought_m.group(1).strip() if thought_m else "",
            "answer":  final.group(1).strip(),
        }

    thought_m = _THOUGHT_RE.search(text)
    action_m  = _ACTION_RE.search(text)
    input_m   = _ACTION_INPUT_RE.search(text)

    if not action_m:
        return {"type": "unparseable", "raw": text}

    try:
        action_input = json.loads(input_m.group(1)) if input_m else {}
    except json.JSONDecodeError:
        action_input = {}

    return {
        "type":         "action",
        "thought":      thought_m.group(1).strip() if thought_m else "",
        "action":       action_m.group(1).strip(),
        "action_input": action_input,
    }


# ── ReAct 核心循环 ─────────────────────────────────────────────────────────────

def run(question: str, max_steps: int = 10, verbose: bool = True, history: list = None) -> Generator[dict, None, None]:
    """
    执行 ReAct 循环，yield 每一步的结构化结果

    参数：
      question: 用户问题
      max_steps: 最大步数
      verbose: 详细输出
      history: 历史对话记录（精简格式，只包含 user-assistant 对）

    每个 yield 的 dict 格式：
      {"step": int, "thought": str, "action": str, "action_input": dict, "observation": str}
    最后一个 yield：
      {"step": int, "thought": str, "type": "final", "answer": str, "history": list}
    """
    from tools import TOOLS_MAP

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    # 追加历史对话记录
    if history:
        messages.extend(history)

    # 追加当前问题
    messages.append({"role": "user", "content": question})

    final_answer = None

    for step in range(1, max_steps + 1):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0,
            stop=["Observation:"],  # 让模型停在调用工具前
        )
        llm_output = response.choices[0].message.content.strip()
        parsed = _parse_step(llm_output)

        if parsed["type"] == "final":
            final_answer = parsed["answer"]
            yield {
                "step":    step,
                "type":    "final",
                "thought": parsed["thought"],
                "answer":  parsed["answer"],
                "history": _build_history(history, question, parsed["answer"]),
            }
            return

        if parsed["type"] == "unparseable":
            final_answer = f"格式解析失败：{llm_output[:200]}"
            yield {
                "step":        step,
                "type":        "error",
                "observation": f"格式解析失败，原始输出：{llm_output[:200]}",
                "history":     _build_history(history, question, final_answer),
            }
            return

        # 执行工具
        tool_name  = parsed["action"]
        tool_args  = parsed["action_input"]
        tool_fn    = TOOLS_MAP.get(tool_name)

        if tool_fn is None:
            observation = f"未知工具 '{tool_name}'，可用工具：{list(TOOLS_MAP.keys())}"
        else:
            try:
                observation = tool_fn(**tool_args)
            except TypeError as e:
                observation = f"工具参数错误: {e}"

        step_result = {
            "step":         step,
            "type":         "action",
            "thought":      parsed["thought"],
            "action":       tool_name,
            "action_input": tool_args,
            "observation":  str(observation),
        }
        yield step_result

        # 将本步结果追加到对话历史
        messages.append({"role": "assistant", "content": llm_output})
        messages.append({
            "role":    "user",
            "content": f"Observation: {observation}\n",
        })

    # 超出最大步数，强制终止
    final_answer = f"已达最大步数 {max_steps}，未能得出最终答案"
    yield {
        "step":    max_steps + 1,
        "type":    "max_steps",
        "answer":  final_answer,
        "history": _build_history(history, question, final_answer),
    }


def _build_history(history: list, question: str, answer: str) -> list:
    """
    构建精简的对话历史（只保留 user-assistant 对，丢弃中间工具调用链）
    每轮结束后只提取：user question + assistant final answer
    """
    new_history = history.copy() if history else []
    new_history.append({"role": "user", "content": question})
    new_history.append({"role": "assistant", "content": answer})
    return new_history


# ── CLI 打印 ──────────────────────────────────────────────────────────────────

COLORS = {
    "thought":  "\033[36m",   # cyan
    "action":   "\033[33m",   # yellow
    "obs":      "\033[32m",   # green
    "final":    "\033[35m",   # magenta
    "error":    "\033[31m",   # red
    "reset":    "\033[0m",
}

def _c(color: str, text: str) -> str:
    return f"{COLORS[color]}{text}{COLORS['reset']}"


def run_and_print(question: str, max_steps: int = 10):
    print(f"\n{'='*60}")
    print(f"问题: {question}")
    print(f"模型: {MODEL}  实现: 手写Prompt解析")
    print('='*60)

    start = time.time()
    step_count = 0

    for step_data in run(question, max_steps=max_steps):
        step_count += 1
        stype = step_data["type"]

        if stype == "action":
            print(f"\n[Step {step_data['step']}]")
            print(_c("thought", f"🧠 Thought: {step_data['thought']}"))
            print(_c("action",  f"🔧 Action:  {step_data['action']}"))
            print(_c("action",  f"   Input:   {json.dumps(step_data['action_input'], ensure_ascii=False)}"))
            print(_c("obs",     f"👁  Obs:     {step_data['observation'][:300]}"))

        elif stype == "final":
            elapsed = time.time() - start
            print(f"\n{'─'*60}")
            if step_data.get("thought"):
                print(_c("thought", f"🧠 Thought: {step_data['thought']}"))
            print(_c("final",  f"\n✅ Final Answer:\n{step_data['answer']}"))
            print(f"\n共 {step_data['step']} 步，耗时 {elapsed:.1f}s")

        elif stype in ("error", "max_steps"):
            print(_c("error", f"\n⚠️  {step_data.get('answer', step_data.get('observation', ''))}"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question",  default="贵州茅台和五粮液2023年的毛利率哪家更高？差多少个百分点？")
    parser.add_argument("--max_steps", type=int, default=10)
    args = parser.parse_args()
    run_and_print(args.question, args.max_steps)



"""
agent.py

"""

import os
import argparse

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

DEFAULT_QUESTION = "贵州茅台和五粮液2023年的毛利率哪家更高？差多少个百分点？"


def run_interactive(mode: str, max_steps: int):
    """
    交互式多轮对话模式：持续接收用户输入，每轮传入历史对话记录
    """
    if mode == "manual":
        from react_manual import run, COLORS
    else:
        from react_function_calling import run, COLORS

    def _c(color: str, text: str) -> str:
        return f"{COLORS[color]}{text}{COLORS['reset']}"

    print(f"\n{'='*60}")
    print(f"ReAct Financial Agent 多轮对话模式")
    print(f"模式: {mode} | 最大步数: {max_steps}")
    print(f"输入 quit 或 exit 退出")
    print('='*60)

    history = None

    while True:
        question = input("\n请输入问题：").strip()
        if question.lower() in ("quit", "exit"):
            print("\n退出程序")
            break
        if not question:
            continue

        print(f"\n{'─'*60}")
        print(f"问题: {question}")
        print('─'*60)

        # 执行 ReAct 循环，传入历史记录
        final_result = None
        for step_data in run(question, max_steps=max_steps, history=history):
            stype = step_data["type"]

            if stype == "action":
                print(f"\n[Step {step_data['step']}]")
                thought = step_data.get("thought", "")
                if thought:
                    print(_c("thought", f"🧠 Thought: {thought}"))
                else:
                    print(_c("thought", "🧠 Thought: （模型内部推理）"))
                print(_c("action",  f"🔧 Action:  {step_data['action']}"))
                print(_c("action",  f"   Input:   {step_data['action_input']}"))
                print(_c("obs",     f"👁  Obs:     {step_data['observation'][:300]}"))

            elif stype == "final":
                print(f"\n{'─'*60}")
                if step_data.get("thought"):
                    print(_c("thought", f"🧠 Thought: {step_data['thought']}"))
                print(_c("final",  f"\n✅ Final Answer:\n{step_data['answer']}"))
                final_result = step_data

            elif stype in ("error", "max_steps"):
                print(_c("error", f"\n⚠️  {step_data.get('answer', step_data.get('observation', ''))}"))
                final_result = step_data

        # 更新历史记录供下一轮使用
        if final_result and "history" in final_result:
            history = final_result["history"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReAct Financial Agent")
    parser.add_argument(
        "--mode", choices=["manual", "fc"], default="manual",
        help="manual=手写Prompt解析版  fc=Function Calling版",
    )
    parser.add_argument("--question",  default=DEFAULT_QUESTION)
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--interactive", "-i", action="store_true", help="多轮对话模式")
    args = parser.parse_args()

    if args.interactive:
        run_interactive(args.mode, args.max_steps)
    else:
        if args.mode == "manual":
            from react_manual import run_and_print
        else:
            from react_function_calling import run_and_print

        run_and_print(args.question, args.max_steps)









