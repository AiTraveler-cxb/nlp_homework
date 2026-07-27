"""
run_demo.py —— 渐进式 Skill Harness 演示入口。

运行方式：
    cd week13harness和skills/homework
    python run_demo.py

演示内容：
  1. 渐进式模式：仅加载索引，命中 Skill 后按需加载完整定义。
  2. 全量加载模式：一次性加载所有 Skill 定义。
  3. 输出两种模式的 Context token 占用对比。
"""

import sys
from pathlib import Path

from harness import SkillHarness, create_demo_data, estimate_tokens


# 保证 Windows Git Bash / cmd 下中文输出不乱码
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")


# 示例查询，覆盖三种 Skill
DEMO_QUERIES = [
    "北京今天天气怎么样？",
    "帮我算一下 (15 + 23) * 4 等于多少",
    "帮我写一份周报",
]


def print_separator(title: str):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def run_mode(mode: str, queries: list[str]) -> list[dict]:
    """运行指定模式，返回每次执行的统计信息。"""
    base_dir = Path(__file__).parent
    create_demo_data(base_dir)

    harness = SkillHarness(base_dir / "skills", mode=mode)
    print(f"\n[模式] {mode}")
    print(f"[索引大小] {estimate_tokens(harness.index_text)} tokens")

    stats = []
    for q in queries:
        result = harness.execute(q)
        print(f"\n  用户：{q}")
        print(f"  命中 Skill：{result.matched_skill or '无'}")
        print(f"  执行步骤：")
        for step in result.steps:
            if step["output"]:
                print(f"    - {step['step']}")
                print(f"      → {step['output']}")
        print(f"  回答：{result.final_answer}")
        print(f"  Context 占用：索引 {result.index_tokens} + Skill {result.loaded_tokens} = {result.peak_tokens} tokens")
        stats.append({
            "query": q,
            "skill": result.matched_skill,
            "peak": result.peak_tokens,
        })

    return stats


def main():
    base_dir = Path(__file__).parent
    create_demo_data(base_dir)

    print_separator("渐进式 Skill Harness 演示")

    # 渐进式模式
    progressive_stats = run_mode("progressive", DEMO_QUERIES)

    # 全量加载模式
    full_stats = run_mode("full", DEMO_QUERIES)

    # 对比汇总
    print_separator("Context 占用对比")
    print(f"{'Query':<40} {'Progressive':>12} {'Full':>12} {'节省':>12}")
    print("-" * 80)
    total_prog = 0
    total_full = 0
    for p, f in zip(progressive_stats, full_stats):
        saving = f["peak"] - p["peak"]
        total_prog += p["peak"]
        total_full += f["peak"]
        print(f"{p['query']:<40} {p['peak']:>12} {f['peak']:>12} {saving:>12}")

    print("-" * 80)
    total_saving = total_full - total_prog
    saving_pct = (total_saving / total_full * 100) if total_full else 0
    print(f"{'总计':<40} {total_prog:>12} {total_full:>12} {total_saving:>12}")
    print(f"\n渐进式模式相比全量加载，Context 总占用减少约 {saving_pct:.1f}%。")


if __name__ == "__main__":
    main()
