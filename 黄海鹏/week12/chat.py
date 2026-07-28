import argparse
import json
import os
import time

from memory import AgentMemory

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

COLORS = {
    "thought": "\033[36m",
    "action": "\033[33m",
    "obs": "\033[32m",
    "final": "\033[35m",
    "error": "\033[31m",
    "reset": "\033[0m",
}


def _c(color: str, text: str) -> str:
    return f"{COLORS[color]}{text}{COLORS['reset']}"


def _print_step(step_data: dict, mode: str):
    stype = step_data["type"]
    if stype == "action":
        print(f"\n[Step {step_data['step']}]")
        if mode == "manual":
            print(_c("thought", f"🧠 Thought: {step_data['thought']}"))
        else:
            print(_c("thought", "🧠 Thought: （模型内部推理，Function Calling 版不可见）"))
        print(_c("action", f"🔧 Action:  {step_data['action']}"))
        print(_c("action", f"   Input:   {json.dumps(step_data['action_input'], ensure_ascii=False)}"))
        print(_c("obs", f"👁  Obs:     {step_data['observation'][:300]}"))
    elif stype == "final":
        print(_c("final", f"\n✅ Final Answer:\n{step_data['answer']}"))
    elif stype in ("error", "max_steps"):
        print(_c("error", f"\n⚠️  {step_data.get('answer', step_data.get('observation', ''))}"))


def _run_single(question: str, mode: str, max_steps: int, memory: AgentMemory):
    if mode == "manual":
        from react_manual import run as react_run
    else:
        from react_function_calling import run as react_run

    start = time.time()
    for step_data in react_run(question, max_steps=max_steps, memory=memory):
        _print_step(step_data, mode)
    elapsed = time.time() - start
    print(f"\n耗时: {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="连续问答式 ReAct Agent CLI")
    parser.add_argument("--mode", choices=["manual", "fc"], default="fc")
    parser.add_argument("--max_steps", type=int, default=10)
    args = parser.parse_args()

    mode = args.mode
    print("欢迎使用 ReAct Agent 连续问答模式")
    print("输入 /quit 退出")
    print("-" * 60)

    memory = AgentMemory()

    while True:
        try:
            question = input("\n你：").strip()
        except EOFError:
            print("\n已退出")
            break

        if not question:
            continue
        if question.lower() in {"/quit", "quit", "exit"}:
            print("再见！")
            break

        print(f"\n{'=' * 60}")
        print(f"问题: {question}")
        print(f"模式: {'手写Prompt解析' if mode == 'manual' else 'Function Calling'}")
        print(f"{'=' * 60}")
        _run_single(question, mode, args.max_steps, memory)


if __name__ == "__main__":
    main()
