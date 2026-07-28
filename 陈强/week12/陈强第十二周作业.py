"""
为 run 函数添加 messages 参数

若 messages=None（兼容单次调用），函数内部自动创建包含 system prompt 和当前 question 的消息列表。

若传入已有列表，则只将当前 question 作为新的 user 消息追加，从而实现多轮上下文保持。

保存最终回复
当模型给出 Final Answer（finish_reason == "stop"）时，将 assistant 消息也追加到 messages，确保下一轮对话能“记住”之前的回答。

新增交互式多轮对话模式

命令行增加 --interactive 选项，进入交互循环，用户可连续提问，所有工具调用和回复都会保留在历史中。

--question 参数支持传入多个问题（用空格分隔），程序会按顺序处理，共享同一段对话历史（相当于自动多轮）。

run_and_print 同步更新
增加 messages 参数，可将外部维护的历史传递给 run。
"""

import os
import json
import time
import logging
import argparse
from typing import Generator, Optional

from openai import OpenAI

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# client = OpenAI(
#     api_key=os.getenv("DASHSCOPE_API_KEY"),
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
# )
# MODEL = os.getenv("AGENT_MODEL", "qwen-max")
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)
MODEL = os.getenv("AGENT_MODEL", "deepseek-v4-flash")

FC_SYSTEM_PROMPT = """你是一个专业的A股金融分析助手。
规则：
- 调用 financial_indicator 或 stock_price 之前，必须先用 company_lookup 获取股票代码
- 数字计算必须使用 calculator 工具，不能心算
- Final Answer 必须引用具体数据来源
- 如果没有合适工具能回答，直接说明原因
"""


def run(question: str, max_steps: int = 10, messages: Optional[list] = None) -> Generator[dict, None, None]:
    """
    执行 Function Calling 版 ReAct 循环，yield 每一步结构化结果

    多轮对话支持：
      - 若 messages 为 None，创建全新的对话历史（单轮模式）
      - 若传入已有消息列表，则仅追加当前问题，实现多轮上下文延续
    """
    from tools import TOOLS_MAP, TOOLS_SCHEMA

    # 初始化或延续对话历史
    if messages is None:
        messages = [
            {"role": "system", "content": FC_SYSTEM_PROMPT},
            {"role": "user",   "content": question},
        ]
    else:
        # 多轮场景：只添加新的用户问题
        messages.append({"role": "user", "content": question})

    for step in range(1, max_steps + 1):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
            temperature=0,
        )
        msg    = response.choices[0].message
        reason = response.choices[0].finish_reason

        # 模型决定直接回答（无工具调用）
        if reason == "stop" or not msg.tool_calls:
            # 将最终 assistant 回复存入历史，以便下一轮对话可见
            messages.append(msg)
            yield {
                "step":   step,
                "type":   "final",
                "thought": "",
                "answer": msg.content or "（模型返回空内容）",
            }
            return

        # 模型请求调用工具
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

            step_result = {
                "step":         step,
                "type":         "action",
                "thought":      "",   # Function Calling 版 Thought 在模型内部，不可见
                "action":       tool_name,
                "action_input": tool_args,
                "observation":  str(observation),
            }
            yield step_result

            messages.append({
                "role":         "tool",
                "tool_call_id": tool_call.id,
                "content":      str(observation),
            })

    yield {
        "step":   max_steps + 1,
        "type":   "max_steps",
        "answer": f"已达最大步数 {max_steps}，未能得出最终答案",
    }


# ── CLI 打印（复用 react_manual 的彩色输出） ───────────────────────────────────

COLORS = {
    "thought": "\033[36m",
    "action":  "\033[33m",
    "obs":     "\033[32m",
    "final":   "\033[35m",
    "error":   "\033[31m",
    "reset":   "\033[0m",
}

def _c(color: str, text: str) -> str:
    return f"{COLORS[color]}{text}{COLORS['reset']}"


def run_and_print(question: str, max_steps: int = 10, messages: Optional[list] = None):
    """运行一次问答并打印详细步骤，支持传入已有的对话历史"""
    print(f"\n{'='*60}")
    print(f"问题: {question}")
    print(f"模型: {MODEL}  实现: Function Calling")
    print('='*60)

    start = time.time()

    # 将外部消息列表传入 run；若是 None，run 会自行创建单轮历史
    for step_data in run(question, max_steps=max_steps, messages=messages):
        stype = step_data["type"]

        if stype == "action":
            print(f"\n[Step {step_data['step']}]")
            # Thought 在 FC 版不可见，显示提示
            print(_c("thought", "🧠 Thought: （模型内部推理，Function Calling 版不可见）"))
            print(_c("action",  f"🔧 Action:  {step_data['action']}"))
            print(_c("action",  f"   Input:   {json.dumps(step_data['action_input'], ensure_ascii=False)}"))
            print(_c("obs",     f"👁  Obs:     {step_data['observation'][:300]}"))

        elif stype == "final":
            elapsed = time.time() - start
            print(f"\n{'─'*60}")
            print(_c("final", f"\n✅ Final Answer:\n{step_data['answer']}"))
            print(f"\n共 {step_data['step']} 步，耗时 {elapsed:.1f}s")

        elif stype in ("error", "max_steps"):
            print(_c("error", f"\n⚠️  {step_data.get('answer', '')}"))


def interactive_loop(max_steps: int = 10):
    """交互式多轮对话，用户可连续提问，上下文自动保持"""
    print("\n🤖 进入交互式多轮对话模式（输入 'exit' 或 'quit' 退出）\n")
    history = [{"role": "system", "content": FC_SYSTEM_PROMPT}]

    while True:
        try:
            question = input("🧑 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见！")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            print("👋 再见！")
            break

        # 运行一轮，共享 history
        run_and_print(question, max_steps=max_steps, messages=history)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--question", nargs="+",
        default=["贵州茅台和五粮液2023年的毛利率哪家更高？差多少个百分点？"],
        help="要提问的问题，可输入多个（空格分隔），按顺序形成多轮对话"
    )
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--interactive", action="store_true",
                        help="进入交互式多轮对话模式")
    args = parser.parse_args()

    if args.interactive:
        interactive_loop(max_steps=args.max_steps)
    else:
        # 支持一次传入多个问题，自动形成多轮对话
        if len(args.question) == 1:
            # 单问题：保持原有调用方式，兼容旧行为
            run_and_print(args.question[0], max_steps=args.max_steps)
        else:
            # 多问题：共享对话历史
            history = [{"role": "system", "content": FC_SYSTEM_PROMPT}]
            for q in args.question:
                run_and_print(q, max_steps=args.max_steps, messages=history)
