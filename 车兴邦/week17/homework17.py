import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import torch

DEFAULT_MODEL = os.getenv("QWEN2_0_5B_PATH", "Qwen/Qwen2-0.5B-Instruct")

SYSTEM_PROMPT = (
    "你是一个算术助手。用户会给你一道算术题，请计算出结果，"
    "并把最终答案放在 <answer> 标签中，例如 <answer>42</answer>。"
    "不要输出其他内容。"
)
TAG_RE = re.compile(r"<answer>\s*(-?\d+)\s*</answer>")
NUM_RE = re.compile(r"-?\d+")

LEVELS = ["L1_add_1digit", "L2_addsub_2digit", "L3_addsub_3digit",
          "L4_mul_1digit", "L5_mul_2x1digit", "L6_mul_2x2digit"]
TRAINED_LEVELS = {"L2_addsub_2digit", "L3_addsub_3digit", "L5_mul_2x1digit"}
LEVEL_MIX = [("L3_addsub_3digit", 0.50), ("L5_mul_2x1digit", 0.25), ("L2_addsub_2digit", 0.25)]


# ---------- 数据与解析 ----------
def make_problem(level, rng):
    if level == "L1_add_1digit":
        a, b = rng.randint(1, 9), rng.randint(1, 9)
        return f"{a} + {b}", a + b
    if level == "L2_addsub_2digit":
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        if rng.random() < 0.5:
            return f"{a} + {b}", a + b
        a, b = max(a, b), min(a, b)
        return f"{a} - {b}", a - b
    if level == "L3_addsub_3digit":
        a, b = rng.randint(100, 999), rng.randint(100, 999)
        if rng.random() < 0.5:
            return f"{a} + {b}", a + b
        a, b = max(a, b), min(a, b)
        return f"{a} - {b}", a - b
    if level == "L4_mul_1digit":
        a, b = rng.randint(2, 9), rng.randint(2, 9)
        return f"{a} × {b}", a * b
    if level == "L5_mul_2x1digit":
        a, b = rng.randint(10, 99), rng.randint(3, 9)
        return f"{a} × {b}", a * b
    if level == "L6_mul_2x2digit":
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        return f"{a} × {b}", a * b
    raise ValueError(level)


def parse_output(text, answer):
    m = TAG_RE.search(text or "")
    fmt_ok = m is not None
    strict_ok = fmt_ok and int(m.group(1)) == answer
    nums = NUM_RE.findall(text or "")
    loose_ok = bool(nums) and int(nums[-1]) == answer
    return fmt_ok, strict_ok, loose_ok


# ---------- trl 0.21 + transformers 5.x 兼容补丁（必须先于 trl 导入执行） ----------
def _apply_compat_patch():
    import trl.import_utils as tiu
    for name in dir(tiu):
        if not (name.startswith("is_") and name.endswith("_available")):
            continue
        fn = getattr(tiu, name)
        if not callable(fn):
            continue
        try:
            val = fn()
        except Exception:
            continue
        if isinstance(val, tuple):
            real = bool(val[0])
            setattr(tiu, name, lambda *a, _r=real, **k: _r)
    from transformers import PreTrainedModel
    if not hasattr(PreTrainedModel, "warnings_issued"):
        PreTrainedModel.warnings_issued = {}


# ---------- 评估（基线/复测共用） ----------
def build_prompts(tokenizer, problems):
    texts = []
    for expr, _ in problems:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"计算：{expr} = ?"}]
        texts.append(tokenizer.apply_chat_template(msgs, tokenize=False,
                                                   add_generation_prompt=True))
    return texts


@torch.no_grad()
def generate(model, tokenizer, texts, do_sample, k=1, batch_size=16, max_new_tokens=64):
    all_outputs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=do_sample,
                             temperature=1.0 if do_sample else None,
                             top_p=1.0 if do_sample else None,
                             num_return_sequences=k if do_sample else 1,
                             pad_token_id=tokenizer.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
        if do_sample:
            all_outputs.extend(decoded[j * k: (j + 1) * k] for j in range(len(batch)))
        else:
            all_outputs.extend(decoded)
    return all_outputs


def _load_for_eval(model_path):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if (Path(model_path) / "adapter_config.json").exists():  # LoRA：基座 + adapter
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(
            DEFAULT_MODEL, dtype=torch.bfloat16, device_map="cuda")
        model = PeftModel.from_pretrained(base, model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    return model, tokenizer


def run_probe(model_path, out_path, n, k, seed):
    model, tokenizer = _load_for_eval(model_path)
    rng = random.Random(seed)
    report = {}
    for level in LEVELS:
        problems = [make_problem(level, rng) for _ in range(n)]
        texts = build_prompts(tokenizer, problems)
        greedy_outs = generate(model, tokenizer, texts, do_sample=False)
        g_fmt = g_loose = 0
        for (_, ans), out in zip(problems, greedy_outs):
            fmt, _, loose = parse_output(out, ans)
            g_fmt += fmt
            g_loose += loose
        sample_outs = generate(model, tokenizer, texts, do_sample=True, k=k)
        s_loose = 0
        loose_pass = 0
        mixed = 0
        for (_, ans), outs in zip(problems, sample_outs):
            rs = [parse_output(o, ans) for o in outs]
            n_l = sum(r[2] for r in rs)
            s_loose += n_l
            loose_pass += n_l > 0
            mixed += 0 < n_l < k
        report[level] = {
            "greedy_format_rate": round(g_fmt / n, 4),
            "greedy_loose_acc": round(g_loose / n, 4),
            "sample_loose_acc": round(s_loose / (n * k), 4),
            "loose_pass@8": round(loose_pass / n, 4),
            "loose_informative_group_rate": round(mixed / n, 4),
            "examples": [{"expr": e, "answer": a, "greedy_output": o}
                         for (e, a), o in list(zip(problems, greedy_outs))[:3]],
        }
        r = report[level]
        print(f"{level:<20} greedy_loose={r['greedy_loose_acc']:.2f} "
              f"fmt={r['greedy_format_rate']:.2f} "
              f"loose_pass@8={r['loose_pass@8']:.2f} "
              f"info={r['loose_informative_group_rate']:.2f}")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"结果已保存：{out_path}")
    return report


# ---------- GRPO 训练 ----------
def build_train_dataset(n, seed):
    from datasets import Dataset
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        r, acc, level = rng.random(), 0.0, LEVEL_MIX[-1][0]
        for lv, p in LEVEL_MIX:
            acc += p
            if r <= acc:
                level = lv
                break
        expr, ans = make_problem(level, rng)
        rows.append({"prompt": [{"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": f"计算：{expr} = ?"}],
                     "answer": ans, "level": level})
    return Dataset.from_list(rows)


def reward_correct(completions, answer, **kwargs):
    return [1.0 if parse_output(c[0]["content"], int(a))[2] else 0.0
            for c, a in zip(completions, answer)]


def reward_format(completions, **kwargs):
    return [0.2 if parse_output(c[0]["content"], 0)[0] else 0.0 for c in completions]


def cmd_train(args):
    _apply_compat_patch()
    from trl import GRPOConfig, GRPOTrainer
    root = Path(args.out_root)
    sfx = f"_{args.tag}" if args.tag else ""
    ckpt = root / (f"grpo_lora_ckpt{sfx}" if args.lora else f"grpo_ckpt{sfx}")
    logp = root / (f"train_log_lora{sfx}.json" if args.lora else f"train_log{sfx}.json")
    dataset = build_train_dataset(args.n_prompts, seed=123)
    peft_config = None
    if args.lora:
        from peft import LoraConfig
        peft_config = LoraConfig(r=16, lora_alpha=32,
                                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    config = GRPOConfig(
        output_dir=str(ckpt),
        model_init_kwargs={"torch_dtype": "bfloat16"},  # 必须 bf16：fp16 一步 NaN
        num_generations=8, beta=0.0, epsilon=0.2, temperature=1.0,
        max_prompt_length=128, max_completion_length=64,
        per_device_train_batch_size=8, gradient_accumulation_steps=4,
        learning_rate=args.lr if not args.lora else 2e-4,
        max_steps=args.max_steps, bf16=True,
        gradient_checkpointing=False,   # transformers 5.x 下会毁掉 generate，必须关
        logging_steps=5, save_strategy="no", report_to=[], seed=42,
        log_completions=args.log_completions,
    )
    trainer = GRPOTrainer(model=args.model, args=config,
                          reward_funcs=[reward_correct, reward_format],
                          train_dataset=dataset, peft_config=peft_config)
    trainer.train()
    ckpt.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(ckpt))
    trainer.processing_class.save_pretrained(str(ckpt))
    with open(logp, "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, ensure_ascii=False, indent=2)
    print(f"\n训练完成。checkpoint: {ckpt}\n训练日志: {logp}")
    if torch.cuda.is_available():
        print(f"GPU 峰值显存: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")


# ---------- 对比 ----------
def cmd_compare(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = Path(args.out_root)
    base = json.load(open(root / "baseline_probe.json", encoding="utf-8"))
    post = json.load(open(root / "post_train_probe.json", encoding="utf-8"))
    print("=" * 92)
    print("前后对比（同 seed=42, 50题/难度；每格=格式率 / greedy正确率 / pass@8）")
    print("=" * 92)
    print(f"{'难度':<20}{'训练集':^6}{'基线':^30}{'训练后':^30}")
    for lv in LEVELS:
        rb, rp = base[lv], post[lv]
        row = (f"{lv:<20}{'√' if lv in TRAINED_LEVELS else '—':^6}"
               + (f"{rb['greedy_format_rate']:.2f}/{rb['greedy_loose_acc']:.2f}/"
                  f"{rb['loose_pass@8']:.2f}").center(30)
               + (f"{rp['greedy_format_rate']:.2f}/{rp['greedy_loose_acc']:.2f}/"
                  f"{rp['loose_pass@8']:.2f}").center(30))
        print(row)
    for lv in sorted(TRAINED_LEVELS):
        print(f"\n[{lv}]")
        for eb, ep in zip(base[lv]["examples"][:2], post[lv]["examples"][:2]):
            print(f"  {eb['expr']} = {eb['answer']}: 前 {eb['greedy_output']!r} -> 后 {ep['greedy_output']!r}")
    # 画训练曲线（有日志时）
    logf = root / "train_log.json"
    if logf.exists():
        logs = [e for e in json.load(open(logf, encoding="utf-8")) if "reward" in e]
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        steps = [e["step"] for e in logs]
        ax[0].plot(steps, [e["rewards/reward_correct/mean"] for e in logs], label="correct")
        ax[0].plot(steps, [e["rewards/reward_format/mean"] for e in logs], "--", label="format")
        ax[0].set_title("reward components"); ax[0].legend(); ax[0].grid(alpha=.3)
        ax[1].plot(steps, [e["entropy"] for e in logs])
        ax[1].set_title("policy entropy"); ax[1].grid(alpha=.3)
        fig.tight_layout()
        p = root / "figures" / "train_curves.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150)
        print(f"\n训练曲线已保存：{p}")


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="GRPO 提升小模型算术能力（学生提交版单文件）")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out_root", default="outputs")
    sub = ap.add_subparsers(dest="mode", required=True)

    p = sub.add_parser("baseline", help="基线摸底")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true")
    p.set_defaults(func=lambda a: run_probe(
        a.model, str(Path(a.out_root) / "baseline_probe.json"),
        n=10 if a.quick else a.n, k=a.k, seed=a.seed))

    p = sub.add_parser("train", help="GRPO 训练")
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--n_prompts", type=int, default=1000)
    p.add_argument("--lr", type=float, default=2e-6)
    p.add_argument("--lora", action="store_true")
    p.add_argument("--tag", default="")
    p.add_argument("--log_completions", action="store_true")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("eval", help="训练后复测（seed 与基线一致）")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="outputs/post_train_probe.json")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=lambda a: run_probe(a.ckpt, a.out, n=a.n, k=a.k, seed=a.seed))

    p = sub.add_parser("compare", help="前后对比 + 曲线")
    p.set_defaults(func=cmd_compare)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
