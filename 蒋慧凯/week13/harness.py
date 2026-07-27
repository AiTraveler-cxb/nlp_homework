"""
harness.py —— 渐进式 Skill Harness 核心实现。

教学重点：
  1. 常驻层只加载 skills/index.md（索引）。
  2. 用户输入命中 trigger 后，才加载对应 Skill 的完整 Markdown。
  3. Skill 执行完毕后释放，Context 恢复到仅索引层。
  4. 支持“渐进式”和“全量加载”两种模式，便于对比 Context 占用。
"""

from __future__ import annotations

import ast
import importlib
import inspect
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ── 工具路由表 ────────────────────────────────────────────────────────────

# 通过 tools/ 包动态加载，但保留一份显式映射供教学说明
def _load_tool_modules():
    """动态导入 tools 包下所有模块，返回 {module_name: module}。"""
    from tools import calculator_tool, file_tool, weather_tool

    return {
        "calculator_tool": calculator_tool,
        "file_tool": file_tool,
        "weather_tool": weather_tool,
    }


TOOL_MODULES = _load_tool_modules()


# ── 数据类 ────────────────────────────────────────────────────────────────


@dataclass
class Skill:
    """解析后的 Skill 对象。"""

    name: str
    description: str
    trigger: list[str]  # 多个触发关键词
    body: str           # frontmatter 之后的完整 Markdown 主体
    raw: str            # 原始文件内容


@dataclass
class ExecutionResult:
    """单次执行结果。"""

    user_input: str
    matched_skill: str | None
    mode: str
    steps: list[dict] = field(default_factory=list)
    final_answer: str = ""
    index_tokens: int = 0
    loaded_tokens: int = 0
    peak_tokens: int = 0
    error: str | None = None


# ── Skill 解析 ─────────────────────────────────────────────────────────────


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """
    解析 Markdown 文件中的 YAML-like frontmatter。

    返回 (frontmatter_dict, body)。
    """
    text = text.strip()
    if not text.startswith("---"):
        raise ValueError("Skill 文件必须以 --- 开头")

    parts = text.split("---", 2)
    if len(parts) < 3:
        raise ValueError("Skill 文件缺少 frontmatter 或主体内容")

    fm_text = parts[1].strip()
    body = parts[2].strip()

    fm: dict[str, str] = {}
    for line in fm_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        fm[key.strip()] = value.strip()

    return fm, body


def load_skill(path: Path) -> Skill:
    """从 Markdown 文件加载并解析 Skill。"""
    raw = path.read_text(encoding="utf-8")
    fm, body = parse_frontmatter(raw)

    required = {"name", "description", "trigger"}
    missing = required - set(fm.keys())
    if missing:
        raise ValueError(f"Skill 文件 {path} 缺少必要字段: {missing}")

    trigger = [t.strip() for t in fm["trigger"].split("|")]
    return Skill(
        name=fm["name"],
        description=fm["description"],
        trigger=trigger,
        body=body,
        raw=raw,
    )


# ── Token 估算（教学简化版）────────────────────────────────────────────────


def estimate_tokens(text: str) -> int:
    """
    教学级 token 估算：中文字符按 1 token，英文单词按 1 token，标点忽略。
    """
    if not text:
        return 0

    # 中文/日文/韩文字符
    cjk = len(re.findall(r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]", text))
    # 英文单词
    words = len(re.findall(r"[a-zA-Z0-9_]+", text))
    return cjk + words


# ── 工具调用解析 ───────────────────────────────────────────────────────────


_STEP_TOOL_RE = re.compile(
    r"调用\s+(?P<module>[a-zA-Z_][a-zA-Z0-9_]*)\.(?P<func>[a-zA-Z_][a-zA-Z0-9_]*)\s*\((?P<args>.*?)\)",
    re.DOTALL,
)


def _extract_arg_value(raw_value: str, context: dict[str, Any]) -> Any:
    """
    从步骤参数字符串中提取实际值。
    支持：
      - 纯字符串字面量（加引号）
      - 引用 context 中的变量名
      - 简单表达式（如 city）
    """
    raw_value = raw_value.strip()
    if not raw_value:
        return ""

    # 字符串字面量
    if (raw_value.startswith('"') and raw_value.endswith('"')) or (
        raw_value.startswith("'") and raw_value.endswith("'")
    ):
        return raw_value[1:-1]

    # 尝试从 context 取变量
    if raw_value in context:
        return context[raw_value]

    # 尝试按 Python 字面量解析（数字、列表等）
    try:
        return ast.literal_eval(raw_value)
    except Exception:
        pass

    # 兜底：按变量名返回，若不存在则返回原字符串
    return context.get(raw_value, raw_value)


def _parse_call_args(args_text: str, context: dict[str, Any]) -> tuple[list[Any], dict[str, Any]]:
    """
    解析函数调用参数字符串，支持位置参数和关键字参数。
    教学级实现：按逗号拆分，简单处理引号内逗号。
    """
    args: list[Any] = []
    kwargs: dict[str, Any] = {}

    if not args_text.strip():
        return args, kwargs

    # 简单分词：不在引号内的逗号才是分隔符
    parts = []
    current = []
    in_quote = None
    for ch in args_text:
        if ch in ('"', "'"):
            if in_quote is None:
                in_quote = ch
            elif in_quote == ch:
                in_quote = None
            current.append(ch)
        elif ch == "," and in_quote is None:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current).strip())

    for part in parts:
        if "=" in part and not part.startswith(("'", '"')):
            # 关键字参数
            key, value = part.split("=", 1)
            kwargs[key.strip()] = _extract_arg_value(value, context)
        else:
            args.append(_extract_arg_value(part, context))

    return args, kwargs


def invoke_tool(step_text: str, context: dict[str, Any]) -> str:
    """
    解析步骤文本中的工具调用并执行。

    若步骤不含工具调用，则返回空字符串表示无需执行。
    """
    match = _STEP_TOOL_RE.search(step_text)
    if not match:
        return ""

    module_name = match.group("module")
    func_name = match.group("func")
    args_text = match.group("args")

    if module_name not in TOOL_MODULES:
        return f"错误：未找到工具模块 {module_name}"

    module = TOOL_MODULES[module_name]
    if not hasattr(module, func_name):
        return f"错误：模块 {module_name} 没有函数 {func_name}"

    func = getattr(module, func_name)
    args, kwargs = _parse_call_args(args_text, context)

    try:
        result = func(*args, **kwargs)
        return str(result)
    except Exception as e:
        return f"工具执行错误：{e}"


# ── 用户输入参数提取（教学简化版）────────────────────────────────────────────


def _extract_city(user_input: str) -> str:
    """从天气查询输入中提取城市名。"""
    # 简单规则：取“天气”或“查天气”前面的名词，或最后一个中文字符串
    user_input = user_input.replace("查一下", "").replace("看一下", "")
    # 优先匹配 "X 天气" 或 "天气 X"
    m = re.search(r"([\u4e00-\u9fff]{2,})\s*[的得地]?\s*天气", user_input)
    if m:
        return m.group(1)
    m = re.search(r"天气\s*([\u4e00-\u9fff]{2,})", user_input)
    if m:
        return m.group(1)
    return "北京"


def _extract_expr(user_input: str) -> str:
    """从计算请求中提取数学表达式。"""
    # 去掉常见前缀后缀与中文字符，保留数字、运算符、括点和函数名
    cleaned = re.sub(r"[计算算一下帮我请等于是多少啊？\?，,。！!]", " ", user_input)

    # 匹配以 '(' 开头、可能跟运算符和数字的完整表达式
    m = re.search(r"\([^()]*(?:\([^()]*\)[^()]*)*\)(?:\s*[\+\-\*/%\^]\s*[\d\.\(]+)*", cleaned)
    if m:
        return m.group(0).strip()

    # 兜底：匹配纯数字和运算符组成的子串
    m = re.search(r"[\d\.\s\+\-\*/%\^\(\)]+(?:\s*[\+\-\*/%\^]\s*[\d\.\(\)]+)*", cleaned)
    if m:
        expr = m.group(0).strip()
        if len(expr) >= 3:
            return expr

    return "1 + 1"


# ── Harness 主类 ─────────────────────────────────────────────────────────


class SkillHarness:
    """渐进式 Skill Harness。"""

    def __init__(self, skills_dir: str | Path, mode: str = "progressive"):
        self.skills_dir = Path(skills_dir)
        self.mode = mode.lower()
        if self.mode not in {"progressive", "full"}:
            raise ValueError("mode 必须是 'progressive' 或 'full'")

        self.index_text = ""
        self.skills: dict[str, Skill] = {}
        self._load_all_skills()

    def _load_all_skills(self):
        """加载索引和全部 Skill 定义。"""
        index_path = self.skills_dir / "index.md"
        if not index_path.exists():
            raise FileNotFoundError(f"Skill 索引不存在: {index_path}")

        self.index_text = index_path.read_text(encoding="utf-8")

        for md_file in sorted(self.skills_dir.glob("*.md")):
            if md_file.name == "index.md":
                continue
            try:
                skill = load_skill(md_file)
                self.skills[skill.name] = skill
            except ValueError as e:
                print(f"[warn] 跳过无效 Skill 文件 {md_file}: {e}")

    def match_skill(self, user_input: str) -> Skill | None:
        """按 trigger 关键词匹配 Skill，返回第一个命中的 Skill。"""
        lowered = user_input.lower()
        for skill in self.skills.values():
            for trigger in skill.trigger:
                if trigger.lower() in lowered:
                    return skill
        return None

    def _build_context(self, skill: Skill | None, user_input: str) -> str:
        """组装当前 context 文本。"""
        parts = ["# System\n", self.index_text]

        if self.mode == "full":
            # 全量模式：一次性塞入所有 Skill 完整定义
            parts.append("\n# All Skills\n")
            for s in self.skills.values():
                parts.append(s.raw)
                parts.append("\n---\n")
        elif skill is not None:
            # 渐进式模式：仅塞入命中的 Skill
            parts.append(f"\n# Active Skill: {skill.name}\n")
            parts.append(skill.raw)

        parts.append(f"\n# User Input\n{user_input}\n")
        return "\n".join(parts)

    def execute(self, user_input: str) -> ExecutionResult:
        """
        执行一次用户请求。

        返回 ExecutionResult，包含匹配到的 Skill、执行步骤、Context token 统计。
        """
        result = ExecutionResult(user_input=user_input, matched_skill=None, mode=self.mode)

        # 1. 常驻索引始终加载
        result.index_tokens = estimate_tokens(self.index_text)

        # 2. 匹配 Skill
        skill = self.match_skill(user_input)
        if skill is None:
            result.final_answer = "未匹配到任何 Skill，请尝试询问天气、计算或周报相关话题。"
            result.peak_tokens = result.index_tokens + estimate_tokens(user_input)
            return result

        result.matched_skill = skill.name

        # 3. 根据模式加载 Context
        context_text = self._build_context(skill, user_input)
        if self.mode == "full":
            result.loaded_tokens = sum(estimate_tokens(s.raw) for s in self.skills.values())
        else:
            result.loaded_tokens = estimate_tokens(skill.raw)
        result.peak_tokens = estimate_tokens(context_text)

        # 4. 执行 Skill 步骤
        # 构建执行上下文：提取用户输入中的关键变量
        exec_context: dict[str, Any] = {"user_input": user_input}
        if skill.name == "weather":
            exec_context["city"] = _extract_city(user_input)
        elif skill.name == "calculator":
            exec_context["expr"] = _extract_expr(user_input)

        # 解析 Steps 区域中的每一行
        steps_text = skill.body
        tool_outputs: list[str] = []

        for line in steps_text.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                # 跳过空行、标题、自检清单项
                continue

            # 只处理带 "调用" 的步骤
            if "调用" in line and "(" in line and ")" in line:
                output = invoke_tool(line, exec_context)
                result.steps.append({"step": line, "output": output})
                if output:
                    tool_outputs.append(output)
            else:
                # 记录为说明性步骤
                result.steps.append({"step": line, "output": ""})

        # 5. 生成最终回答（教学简化：用规则拼接工具输出）
        result.final_answer = self._compose_answer(skill, user_input, tool_outputs, exec_context)

        return result

    def _compose_answer(
        self,
        skill: Skill,
        user_input: str,
        tool_outputs: list[str],
        exec_context: dict[str, Any],
    ) -> str:
        """根据 Skill 类型和工具输出组织自然语言回答。"""
        if skill.name == "weather":
            city = exec_context.get("city", "")
            if tool_outputs:
                return f"[{skill.name}] {tool_outputs[0]}"
            return f"[{skill.name}] 未能获取 {city} 的天气信息"

        if skill.name == "calculator":
            if tool_outputs:
                return f"[{skill.name}] {tool_outputs[0]}"
            return f"[{skill.name}] 未能完成计算"

        if skill.name == "weekly_report":
            if len(tool_outputs) >= 2:
                tasks_text = tool_outputs[0]
                done = tasks_text.count("- [x]")
                todo = tasks_text.count("- [ ]")
                return (
                    f"[{skill.name}] 本周共 {done + todo} 项任务，已完成 {done} 项，待完成 {todo} 项。"
                    f"\n{tool_outputs[-1]}"
                )
            return f"[{skill.name}] 周报生成完成"

        return f"[{skill.name}] 执行完成，工具输出：{tool_outputs}"


# ── 便捷函数 ───────────────────────────────────────────────────────────────


def create_demo_data(base_dir: Path):
    """为 weekly_report Skill 准备示例任务文件。"""
    data_dir = base_dir / "data"
    data_dir.mkdir(exist_ok=True)
    tasks_path = data_dir / "weekly_tasks.md"
    if not tasks_path.exists():
        tasks_path.write_text(
            "# 本周任务\n\n"
            "- [x] 完成 Agent 需求评审\n"
            "- [x] 设计四层记忆模型\n"
            "- [ ] 跑通 agent_memory_system demo\n"
            "- [ ] 编写渐进式 Skill Harness 作业\n"
            "- [x] 整理 Week 13 学习笔记\n",
            encoding="utf-8",
        )
