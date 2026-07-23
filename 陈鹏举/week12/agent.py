if __name__ == "__main__":
    parser.add_argument("--interactive", action="store_true", help="进入多轮交互模式")
    # ... 其他参数
    args = parser.parse_args()

    if args.interactive:
        if args.mode == "manual":
            from react_manual import run, SYSTEM_PROMPT
        else:
            from react_function_calling import run, FC_SYSTEM_PROMPT

        history = [{"role": "system", "content": SYSTEM_PROMPT if args.mode == "manual" else FC_SYSTEM_PROMPT}]
        print("进入交互模式，输入 'exit' 退出")
        while True:
            q = input("\n你: ")
            if q.lower() in ("exit", "quit"):
                break
            print("\nAgent:")
            for step_data in run(q, max_steps=args.max_steps, history=history):
                # 根据 step_data["type"] 打印（可复用 run_and_print 的展示逻辑）
                # 为了简洁，此处只打印最终答案
                if step_data["type"] == "final":
                    print(step_data["answer"])
                    break
                elif step_data["type"] == "action":
                    print(f"[工具调用] {step_data['action']} => {step_data['observation'][:100]}...")
    else:
        # 原有单次执行逻辑
        ...
