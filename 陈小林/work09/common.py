import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    print(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        dtype=torch.float16,
        device="cpu")

    model.eval()
    with torch.no_grad():
        messages = [{"role": "user", "content": "讲一下龟兔赛跑的故事"}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer.tokenize(text)
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.8,
            top_p=0.95,
            do_sample=True,
        )

        # 解码
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(response)