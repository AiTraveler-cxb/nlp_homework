import time

from mlx_lm import load, generate

if __name__ == "__main__":
    model, tokenizer = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")

    messages = [{"role": "user", "content": "讲一下龟兔赛跑的故事，要有结局"}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # 旧版 generate 只接受最基础的参数
    start = time.time()
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=2048,
        # 不要传 temperature, top_p 等参数
    )
    end = time.time()
    print(f'消耗时间：{end - start}，响应：{response}')