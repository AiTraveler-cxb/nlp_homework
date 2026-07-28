
from __future__ import annotations

import json
import time
from typing import Generator


class _BaseConversation:
    """多轮对话基类：维护跨轮次的 messages 与轮次历史记录（延迟加载依赖）"""

    def __init__(self, max_steps: int = 10):
        self.max_steps = max_steps
        # messages 在首次 ask 时由 _load() 初始化，整个会话生命周期内持续累积
        self.messages: list[dict] | None = None
        self.rounds: list[dict] = []
        self.model_name: str = ""
        self._loaded = False

    # ── 延迟加载 ──────────────────────────────────────────────────────────
    def _load(self):
        """子类实现：按需 import 依赖模块并初始化 messages / system prompt"""
        raise NotImplementedError

    def _ensure_loaded(self):
        if not self._loaded:
            self._load()
            self._loaded = True

    # ── 公共 API ──────────────────────────────────────────────────────────
    def reset(self):
        """清空对话历史，开启全新会话（保留 system prompt）"""
        self._ensure_loaded()
        self.messages = [self.messages[0]]
        self.rounds = []

    def ask(self, question: str) -> Generator[dict, None, None]:
        """
        提出一轮问题，yield 每一步结构化结果。

        yield 的 dict 格式与单轮 react_manual.run() / react_function_calling.run()
        保持一致，便于复用相同的打印/评估逻辑。
        结果会沉淀到 self.messages，供后续轮次引用。
        """
        self._ensure_loaded()

        round_idx = len(self.rounds) + 1
        self.messages.append({"role": "user", "content": question})

        steps: list[dict] = []
        start = time.time()
        for step_data in self._run_round():
            steps.append(step_data)
            yield step_data

        self.rounds.append({
            "round":     round_idx,
            "question":  question,
            "steps":     steps,
            "elapsed_s": round(time.time() - start, 1),
        })

    # ── 子类实现 ──────────────────────────────────────────────────────────
    def _run_round(self) -> Generator[dict, None, None]:
        raise NotImplementedError


class ManualConversation(_BaseConversation):
    """手写 Prompt 解析版多轮对话（Thought 完全可见）"""

    def _load(self):
        # 延迟 import：避免本模块 import 时即创建 DashScope 客户端
        import react_manual as rm
        from tools import TOOLS_MAP
        self._rm = rm
        self._tools_map = TOOLS_MAP
        self.model_name = rm.MODEL
        self.messages = [{"role": "system", "content": rm.SYSTEM_PROMPT}]

    def _run_round(self) -> Generator[dict, None, None]:
        rm = self._rm
        for step in range(1, self.max_steps + 1):
            response = rm.client.chat.completions.create(
                model=rm.MODEL,
                messages=self.messages,
                temperature=0,
                stop=["Observation:"],   # 让模型停在调用工具前
            )
            llm_output = response.choices[0].message.content.strip()
            parsed = rm._parse_step(llm_output)

            if parsed["type"] == "final":
                # ★ 关键：把最终回答存入 messages，下一轮可引用
                self.messages.append({"role": "assistant", "content": llm_output})
                yield {
                    "step":    step,
                    "type":    "final",
                    "thought": parsed["thought"],
                    "answer":  parsed["answer"],
                }
                return

            if parsed["type"] == "unparseable":
                # 格式异常：兜底追加 assistant 消息，保证下一轮上下文连贯
                self.messages.append({
                    "role": "assistant",
                    "content": f"(上一轮因格式解析失败终止: {llm_output[:200]})",
                })
                yield {
                    "step":        step,
                    "type":        "error",
                    "observation": f"格式解析失败，原始输出：{llm_output[:200]}",
                }
                return

            # 执行工具
            tool_name = parsed["action"]
            tool_args = parsed["action_input"]
            tool_fn   = self._tools_map.get(tool_name)

            if tool_fn is None:
                observation = f"未知工具 '{tool_name}'，可用工具：{list(self._tools_map.keys())}"
            else:
                try:
                    observation = tool_fn(**tool_args)
                except TypeError as e:
                    observation = f"工具参数错误: {e}"

            yield {
                "step":         step,
                "type":         "action",
                "thought":      parsed["thought"],
                "action":       tool_name,
                "action_input": tool_args,
                "observation":  str(observation),
            }

            # 追加本步到 messages，形成上下文记忆（与单轮逻辑一致）
            self.messages.append({"role": "assistant", "content": llm_output})
            self.messages.append({
                "role":    "user",
                "content": f"Observation: {observation}\n",
            })

        # 超出最大步数，强制终止并兜底，避免下一轮上下文断裂
        self.messages.append({
            "role": "assistant",
            "content": f"(已达最大步数 {self.max_steps}，未能得出最终答案)",
        })
        yield {
            "step":   self.max_steps + 1,
            "type":   "max_steps",
            "answer": f"已达最大步数 {self.max_steps}，未能得出最终答案",
        }


class FCConversation(_BaseConversation):
    """Function Calling 版多轮对话（Thought 不可见）"""

    def _load(self):
        # 延迟 import：避免本模块 import 时即创建 DeepSeek 客户端
        import react_function_calling as rfc
        from tools import TOOLS_MAP, TOOLS_SCHEMA
        self._rfc = rfc
        self._tools_map = TOOLS_MAP
        self._tools_schema = TOOLS_SCHEMA
        self.model_name = rfc.MODEL
        self.messages = [{"role": "system", "content": rfc.FC_SYSTEM_PROMPT}]

    def _run_round(self) -> Generator[dict, None, None]:
        rfc = self._rfc
        for step in range(1, self.max_steps + 1):
            response = rfc.client.chat.completions.create(
                model=rfc.MODEL,
                messages=self.messages,
                tools=self._tools_schema,
                tool_choice="auto",
                temperature=0,
            )
            msg    = response.choices[0].message
            reason = response.choices[0].finish_reason

            # 模型决定直接回答（无工具调用）→ 本轮结束
            if reason == "stop" or not msg.tool_calls:
                # ★ 关键：把最终回答存入 messages，下一轮可引用
                self.messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                })
                yield {
                    "step":   step,
                    "type":   "final",
                    "thought": "",   # Function Calling 版 Thought 在模型内部，不可见
                    "answer": msg.content or "（模型返回空内容）",
                }
                return

            # 模型请求调用工具
            self.messages.append(msg)

            for tool_call in msg.tool_calls:
                tool_name = tool_call.function.name
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {}

                tool_fn = self._tools_map.get(tool_name)
                if tool_fn is None:
                    observation = f"未知工具 '{tool_name}'"
                else:
                    try:
                        observation = tool_fn(**tool_args)
                    except TypeError as e:
                        observation = f"工具参数错误: {e}"

                yield {
                    "step":         step,
                    "type":         "action",
                    "thought":      "",
                    "action":       tool_name,
                    "action_input": tool_args,
                    "observation":  str(observation),
                }
                self.messages.append({
                    "role":         "tool",
                    "tool_call_id": tool_call.id,
                    "content":      str(observation),
                })

        # 超出最大步数兜底
        self.messages.append({
            "role": "assistant",
            "content": f"(已达最大步数 {self.max_steps}，未能得出最终答案)",
        })
        yield {
            "step":   self.max_steps + 1,
            "type":   "max_steps",
            "answer": f"已达最大步数 {self.max_steps}，未能得出最终答案",
        }


def create_conversation(mode: str = "manual", max_steps: int = 10) -> _BaseConversation:
    """
    工厂函数：按模式创建多轮对话实例（仅创建对象，不加载依赖）

    mode:
      "manual" — 手写 Prompt 解析版（Thought 可见）
      "fc"     — Function Calling 版（Thought 不可见）
    """
    if mode == "manual":
        return ManualConversation(max_steps=max_steps)
    if mode == "fc":
        return FCConversation(max_steps=max_steps)
    raise ValueError(f"未知模式: {mode}，可选 manual / fc")
