"""
多轮对话版 ReAct Agent

运行：
    python agent.py

命令：
    clear  清空对话记忆
    exit   退出程序
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from react_manual import run_and_print

def main():
    history = []
    print("输入 clear 清空记忆，输入 exit 退出")
    while True:
        question = input("\n用户：").strip()
        if question.lower() == "exit":
            print("已退出。")
            break
        if question.lower() == "clear":
            history.clear()
            print("对话记忆已清空。")
            continue
        answer = run_and_print(
            question=question,
            max_steps=10,
            history=history,
        )
        if answer:
            history.append({
                "role": "user",
                "content": question,
            })
            history.append({
                "role": "assistant",
                "content": answer,
            })
if __name__ == "__main__":
    main()