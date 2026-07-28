import os
import sys
import json
import argparse
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).parent))

from conversation import create_conversation

# 复用与 react_manual 一致的彩色输出（本地定义，避免 import react_manual
# 时触发 LLM 客户端创建，保证 CLI 在未配置 key 时也能启动展示帮助）
COLORS = {
    "thought": "\033[36m",   # cyan
    "action":  "\033[33m",   # yellow
    "obs":     "\033[32m",   # green
    "final":   "\033[35m",   # magenta
    "error":   "\033[31m",   # red
    "reset":   "\033[0m",
}

def _c(color: str, text: str) -> str:
    return f"{COLORS[color]}{text}{COLORS['reset']}"


def _print_step(step_data: dict, mode: str):
    stype = step_data["type"]

    if stype == "action":
        thought = step_data.get("thought", "")
        if thought:
            print(_c("thought", f"  🧠 Thought: {thought}"))
        elif mode == "fc":
            print(_c("thought", "  🧠 Thought: （模型内部推理，Function Calling 版不可见）"))
        print(_c("action", f"  🔧 Action:  {step_data['action']}"))
        print(_c("action", f"     Input:   {json.dumps(step_data['action_input'], ensure_ascii=False)}"))
        print(_c("obs",     f"  👁  Obs:     {step_data['observation'][:300]}"))

    elif stype == "final":
        if step_data.get("thought"):
            print(_c("thought", f"  🧠 Thought: {step_data['thought']}"))
        print(_c("final", f"  ✅ Final Answer:\n{step_data['answer']}"))

    elif stype in ("error", "max_steps"):
        print(_c("error", f"  ⚠️  {step_data.get('answer', step_data.get('observation', ''))}"))


def main():
    parser = argparse.ArgumentParser(description="ReAct 多轮对话 CLI")
    parser.add_argument("--mode", choices=["manual", "fc"], default="manual",
                        help="manual=手写Prompt解析版  fc=Function Calling版")
    parser.add_argument("--max_steps", type=int, default=10)
    args = parser.parse_args()

    conv = create_conversation(args.mode, args.max_steps)

    print(f"\n{'='*60}")
    print(f"ReAct 多轮对话  | 实现: {args.mode}  | 模型: {conv.model_name}")
    print("直接输入问题即可追问，上下文自动保留。")
    print("命令: :reset 清空历史  :history 查看轮次  :quit 退出")
    print('='*60)

    while True:
        try:
            question = input("\n🙋 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not question:
            continue

        if question in (":quit", ":q", ":exit"):
            print("再见！")
            break

        if question == ":reset":
            conv.reset()
            print(_c("obs", "  已清空对话历史，开启新会话。"))
            continue

        if question == ":history":
            if not conv.rounds:
                print("  （暂无历史）")
                continue
            for r in conv.rounds:
                print(f"  [轮{r['round']}] {r['question']}  "
                      f"({len(r['steps'])}步, {r['elapsed_s']}s)")
            continue

        # 正常提问：上下文自动累积
        print(_c("action", "🤖 Agent:"))
        for step_data in conv.ask(question):
            _print_step(step_data, args.mode)


if __name__ == "__main__":
    main()
