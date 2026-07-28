"""
统一入口：切换手写版 / Function Calling 版 ReAct Agent

使用方式：
  python agent.py
  python agent.py --mode manual   --question "茅台2023年毛利率是多少？"
  python agent.py --mode fc       --question "五粮液近一年股价涨跌幅？"
  python agent.py --mode manual   --question "..." --max_steps 8

环境变量：
  DASHSCOPE_API_KEY  必填
  AGENT_MODEL        默认 qwen-max，可换 deepseek-v3 等
"""

import os
import argparse

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

DEFAULT_QUESTION = "贵州茅台和五粮液2023年的毛利率哪家更高？差多少个百分点？"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReAct Financial Agent")
    parser.add_argument(
        "--mode", choices=["manual", "fc"], default="manual",
        help="manual=手写Prompt解析版  fc=Function Calling版",
    )
    parser.add_argument("--question",  default=DEFAULT_QUESTION)
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument(
        "--interactive", "-i", action="store_true",
        help="多轮对话模式：连续提问，自动携带历史上下文",
    )
    args = parser.parse_args()

    if args.mode == "manual":
        from react_manual import run_and_print
    else:
        from react_function_calling import run_and_print

    if not args.interactive:
        run_and_print(args.question, args.max_steps)
    else:
        history = []
        print(f"\n多轮对话模式（{args.mode}），输入 exit / quit / 退出 结束。\n")
        while True:
            try:
                question = input(f"[第{len(history)//2 + 1}轮] 你> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not question:
                continue
            if question.lower() in ("exit", "quit", "q", "退出"):
                print("对话结束。")
                break
            answer = run_and_print(question, args.max_steps, history=history)
            if answer:
                history.append({"role": "user", "content": question})
                history.append({"role": "assistant", "content": answer})
