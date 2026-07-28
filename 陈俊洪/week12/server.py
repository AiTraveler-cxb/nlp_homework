"""
多轮对话增强层：
为现有 ReAct Agent 补上跨轮对话记忆。

问题背景：
  react_manual.run() 和 react_function_calling.run() 每次都从
  messages = [system, question] 全新开始，没有跨轮记忆。用户问第二个
  问题时，Agent 无法理解 "它""这家公司""再对比一下" 这类指代和追问。

使用方式：
  # 交互式多轮对话
  python conversation.py
  python conversation.py --mode fc

  # 代码里调用
  from conversation import ChatSession
  chat = ChatSession(mode="manual")
  print(chat.ask("贵州茅台2023年毛利率是多少？"))
  print(chat.ask("那五粮液呢？"))          # 自动理解「那...呢」是在问毛利率
  print(chat.ask("它俩差多少个百分点？"))   # 自动理解「它俩」指茅台和五粮液

"""

import os
import json
import time
import argparse
from typing import Generator, Optional

from openai import OpenAI

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ── 复用与 react_manual 完全相同的 LLM 配置（用于问题改写/摘要）─────────────────
_client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
_MODEL = os.getenv("AGENT_MODEL", "qwen-max")

# 历史轮数超过该阈值后，把较早的轮次压缩成一段摘要
_SUMMARIZE_AFTER_TURNS = 6
# 改写/摘要这类轻量任务的最大 token，够用即可
_REWRITE_MAX_TOKENS = 512


# ── 问题改写 Prompt ─────────────────────────────────────────────────────────────
_REWRITE_SYSTEM = """你是一个「问题改写器」。给定多轮对话历史和用户的最新一句话，
请把最新问题改写成一个「不依赖上下文也能独立理解」的完整问题。

要求：
- 消解所有指代词：把「它/它俩/这家公司/那/上面那个」等替换为具体的公司名、指标名等
- 补全省略成分：如果最新问题是「那五粮液呢？」，要根据上文补成「五粮液2023年的毛利率是多少？」
- 保持用户的原始意图，不要自行扩大或缩小问题范围
- 不要回答问题，只输出改写后的问题本身，不要任何解释、前缀或引号
- 如果最新问题本身已经自包含、无需改写，原样输出即可
"""

# ── 摘要 Prompt ─────────────────────────────────────────────────────────────────
_SUMMARIZE_SYSTEM = """请把下面这段多轮金融问答对话压缩成简洁的中文摘要，
保留关键实体（公司名、股票代码）、已查到的具体数值（指标/股价/日期）和已得出的结论。
用要点列举，不要丢失数字。只输出摘要正文。
"""


class ConversationMemory:
    """跨轮对话记忆：原始轮次 + 早期轮次的滚动摘要 + 事实卡片。"""

    def __init__(self, summarize_after: int = _SUMMARIZE_AFTER_TURNS):
        self.turns: list[dict] = []       # [{"question": ..., "answer": ...}, ...]
        self.summary: str = ""            # 早期轮次压缩后的摘要
        self.facts: list[str] = []        # 沉淀的事实卡片（股票代码/指标数值等）
        self._summarize_after = summarize_after

    # ---- 写入 ----
    def add_turn(self, question: str, answer: str) -> None:
        self.turns.append({"question": question, "answer": answer})
        self._maybe_summarize()

    def add_facts(self, facts: list[str]) -> None:
        for f in facts:
            if f and f not in self.facts:
                self.facts.append(f)

    # ---- 读取 ----
    def is_empty(self) -> bool:
        return not self.turns and not self.summary

    def context_text(self) -> str:
        """把摘要 + 近几轮 + 事实卡片拼成一段供改写器阅读的上下文。"""
        parts: list[str] = []
        if self.summary:
            parts.append(f"【早前对话摘要】\n{self.summary}")
        if self.facts:
            parts.append("【已知事实】\n" + "\n".join(f"- {f}" for f in self.facts))
        if self.turns:
            recent = []
            for i, t in enumerate(self.turns, 1):
                recent.append(f"用户：{t['question']}\n助手：{t['answer']}")
            parts.append("【最近对话】\n" + "\n\n".join(recent))
        return "\n\n".join(parts)

    # ---- 内部：滚动摘要 ----
    def _maybe_summarize(self) -> None:
        if len(self.turns) <= self._summarize_after:
            return
        # 把最早的一半轮次压进摘要，保留最近一半为原文
        keep = self._summarize_after // 2
        old, self.turns = self.turns[:-keep], self.turns[-keep:]

        dialog = "\n\n".join(
            f"用户：{t['question']}\n助手：{t['answer']}" for t in old
        )
        prior = f"已有摘要：\n{self.summary}\n\n新增对话：\n" if self.summary else ""
        try:
            resp = _client.chat.completions.create(
                model=_MODEL,
                messages=[
                    {"role": "system", "content": _SUMMARIZE_SYSTEM},
                    {"role": "user", "content": prior + dialog},
                ],
                temperature=0,
                max_tokens=_REWRITE_MAX_TOKENS,
            )
            self.summary = resp.choices[0].message.content.strip()
        except Exception as e:  # 摘要失败不影响主流程，退化为拼接
            self.summary = (self.summary + "\n" + dialog).strip()
            _ = e


def _rewrite_question(memory: ConversationMemory, question: str) -> str:
    """把用户最新问题改写为自包含问题；首轮或改写失败时原样返回。"""
    if memory.is_empty():
        return question
    try:
        resp = _client.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": _REWRITE_SYSTEM},
                {"role": "user", "content":
                    f"对话历史：\n{memory.context_text()}\n\n"
                    f"用户最新问题：{question}\n\n改写后的独立问题："},
            ],
            temperature=0,
            max_tokens=_REWRITE_MAX_TOKENS,
        )
        rewritten = resp.choices[0].message.content.strip()
        return rewritten or question
    except Exception:
        return question


def _extract_facts(steps: list[dict]) -> list[str]:
    """从本轮的工具观测里沉淀事实卡片（股票代码、指标数值等）。"""
    facts: list[str] = []
    for s in steps:
        if s.get("type") != "action":
            continue
        action = s.get("action", "")
        obs = str(s.get("observation", "")).strip()
        if not obs:
            continue
        args = s.get("action_input", {})
        # 只沉淀最有复用价值的几类工具，截断过长观测
        if action in ("company_lookup", "financial_indicator", "stock_price"):
            arg_str = json.dumps(args, ensure_ascii=False)
            facts.append(f"{action}{arg_str} → {obs[:120]}")
    return facts


class ChatSession:
    """
    多轮对话会话：包裹原始 run() 生成器，对外暴露 ask() / stream()。

    mode="manual" 走 react_manual.run，mode="fc" 走 react_function_calling.run。
    """

    def __init__(self, mode: str = "manual", max_steps: int = 10,
                 use_facts: bool = True):
        if mode == "manual":
            from react_manual import run as _run
        elif mode == "fc":
            from react_function_calling import run as _run
        else:
            raise ValueError(f"未知 mode: {mode}，应为 'manual' 或 'fc'")
        self._run = _run
        self.mode = mode
        self.max_steps = max_steps
        self.use_facts = use_facts
        self.memory = ConversationMemory()

    def stream(self, question: str) -> Generator[dict, None, None]:
        """
        执行一轮对话，yield 每一步（格式与原始 run() 完全一致），
        并额外在最前面 yield 一条 {"type": "rewrite", ...} 展示问题改写结果。
        轮末自动把 (原始问题, 最终答案) 写入记忆。
        """
        rewritten = _rewrite_question(self.memory, question)
        if rewritten != question:
            yield {"type": "rewrite", "original": question, "rewritten": rewritten}

        steps: list[dict] = []
        final_answer = ""
        for step_data in self._run(rewritten, max_steps=self.max_steps):
            steps.append(step_data)
            if step_data.get("type") == "final":
                final_answer = step_data.get("answer", "")
            elif step_data.get("type") in ("max_steps", "error"):
                final_answer = step_data.get("answer") or step_data.get("observation", "")
            yield step_data

        # 记忆用原始问题（保留用户口吻），答案用最终结论
        self.memory.add_turn(question, final_answer)
        if self.use_facts:
            self.memory.add_facts(_extract_facts(steps))

    def ask(self, question: str) -> str:
        """便捷同步接口：跑完一轮，只返回最终答案字符串。"""
        answer = ""
        for step_data in self.stream(question):
            if step_data.get("type") == "final":
                answer = step_data.get("answer", "")
            elif step_data.get("type") in ("max_steps", "error"):
                answer = step_data.get("answer") or step_data.get("observation", "")
        return answer

    def reset(self) -> None:
        """清空对话记忆，开始新会话。"""
        self.memory = ConversationMemory()


# ── 交互式 REPL ────────────────────────────────────────────────────────────────
_COLORS = {
    "thought": "\033[36m", "action": "\033[33m", "obs": "\033[32m",
    "final": "\033[35m", "error": "\033[31m", "rewrite": "\033[90m",
    "user": "\033[1m", "reset": "\033[0m",
}

def _c(color: str, text: str) -> str:
    return f"{_COLORS[color]}{text}{_COLORS['reset']}"


def _print_step(step_data: dict) -> None:
    stype = step_data["type"]
    if stype == "rewrite":
        print(_c("rewrite", f"  ↻ 改写为独立问题：{step_data['rewritten']}"))
    elif stype == "action":
        print(f"\n[Step {step_data['step']}]")
        if step_data.get("thought"):
            print(_c("thought", f"🧠 Thought: {step_data['thought']}"))
        print(_c("action", f"🔧 Action:  {step_data['action']}"))
        print(_c("action", f"   Input:   {json.dumps(step_data['action_input'], ensure_ascii=False)}"))
        print(_c("obs", f"👁  Obs:     {str(step_data['observation'])[:300]}"))
    elif stype == "final":
        print(_c("final", f"\n✅ {step_data['answer']}"))
    elif stype in ("error", "max_steps"):
        print(_c("error", f"\n⚠️  {step_data.get('answer', step_data.get('observation', ''))}"))


def repl(mode: str = "manual", max_steps: int = 10, verbose: bool = True):
    chat = ChatSession(mode=mode, max_steps=max_steps)
    print(f"\n{'='*60}")
    print(f"多轮金融对话 Agent  |  模型: {_MODEL}  |  实现: {mode}")
    print("输入问题开始对话；:reset 清空记忆；:quit 退出")
    print('='*60)

    while True:
        try:
            question = input(_c("user", "\n你 > ")).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if not question:
            continue
        if question in (":quit", ":q", ":exit"):
            print("再见！")
            break
        if question == ":reset":
            chat.reset()
            print(_c("rewrite", "  已清空对话记忆。"))
            continue

        start = time.time()
        for step_data in chat.stream(question):
            if verbose or step_data["type"] in ("final", "error", "max_steps", "rewrite"):
                _print_step(step_data)
        print(_c("rewrite", f"  （耗时 {time.time() - start:.1f}s，历史 {len(chat.memory.turns)} 轮）"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多轮对话增强版 ReAct Agent")
    parser.add_argument("--mode", choices=["manual", "fc"], default="manual",
                        help="manual=手写Prompt版  fc=Function Calling版")
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--quiet", action="store_true",
                        help="只显示最终答案，隐藏中间推理步骤")
    args = parser.parse_args()
    repl(mode=args.mode, max_steps=args.max_steps, verbose=not args.quiet)
