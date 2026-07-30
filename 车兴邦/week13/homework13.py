from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except ImportError:  # 允许无 openai 包时仍可用关键词兜底演示
    OpenAI = None

PROJECT_ROOT = Path(__file__).parent
SKILLS_ROOTS = [PROJECT_ROOT.parent / "skills", PROJECT_ROOT.parent / ".cursor" / "skills"]
MAX_INLINE_FILE_CHARS = 24_000


# ============================================================
# 1. LLM 配置：失败时不影响关键词兜底演示
# ============================================================

PROVIDERS = {
    "deepseek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-v4-flash",
    },
    "qwen": {
        "api_key_env": "DASHSCOPE_API_KEY",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
    },
}


def get_chat_client():
    if OpenAI is None:
        raise RuntimeError("未安装 openai 包，跳过 LLM 调用")
    provider = os.getenv("LLM_PROVIDER", "deepseek").lower()
    cfg = PROVIDERS.get(provider, PROVIDERS["deepseek"])
    api_key = os.getenv(cfg["api_key_env"])
    if not api_key:
        raise RuntimeError(f"缺少环境变量 {cfg['api_key_env']}，跳过 LLM 调用")
    return OpenAI(api_key=api_key, base_url=cfg["base_url"]), cfg["model"]


# ============================================================
# 2. 数据结构
# ============================================================

@dataclass
class SkillMeta:
    name: str
    description: str
    path: Path
    version: str = ""


@dataclass
class SkillRunResult:
    request: str
    matched_skill: str | None
    loaded_files: list[str] = field(default_factory=list)
    assistant_plan: str = ""
    executed_commands: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "request": self.request,
            "matched_skill": self.matched_skill,
            "loaded_files": self.loaded_files,
            "assistant_plan": self.assistant_plan,
            "executed_commands": self.executed_commands,
            "artifacts": self.artifacts,
            "error": self.error,
        }


# ============================================================
# 3. Catalog：只读取 frontmatter
# ============================================================

class SkillCatalog:
    """轻量技能目录：启动时只读 SKILL.md 的 frontmatter。"""

    def __init__(self, roots: list[Path] | None = None):
        self.roots = roots or SKILLS_ROOTS
        self.skills: list[SkillMeta] = []

    def scan(self) -> list[SkillMeta]:
        found: dict[str, SkillMeta] = {}
        for root in self.roots:
            if not root.exists():
                continue
            for skill_md in root.glob("*/SKILL.md"):
                meta = self._read_frontmatter(skill_md)
                if not meta:
                    continue
                # skills/ 优先，.cursor/skills 作为兼容备份，避免重复。
                old = found.get(meta.name)
                if old is None or ".cursor" in old.path.parts:
                    found[meta.name] = meta
        self.skills = sorted(found.values(), key=lambda x: x.name)
        return self.skills

    def _read_frontmatter(self, path: Path) -> SkillMeta | None:
        text = path.read_text(encoding="utf-8")
        if not text.startswith("---"):
            return None
        end = text.find("\n---", 3)
        if end == -1:
            return None
        frontmatter = text[3:end].strip()
        data = parse_simple_yaml_frontmatter(frontmatter)
        return SkillMeta(
            name=data.get("name") or path.parent.name,
            description=data.get("description", ""),
            version=data.get("version", ""),
            path=path,
        )


def parse_simple_yaml_frontmatter(text: str) -> dict[str, str]:
    """
    简化版 YAML frontmatter 解析器。
    为了单文件作业不依赖 PyYAML，只支持本课程 SKILL.md 用到的 key: value 和 >- 多行描述。
    """
    data: dict[str, str] = {}
    current_key = ""
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line:
            continue
        if re.match(r"^[A-Za-z_][\w-]*:\s*", line):
            key, _, val = line.partition(":")
            current_key = key.strip()
            val = val.strip().strip('"\'')
            data[current_key] = "" if val in {">-", "|", "|-"} else val
        elif current_key:
            data[current_key] = (data[current_key] + " " + line.strip()).strip()
    return data


# ============================================================
# 4. Progressive Harness
# ============================================================

class ProgressiveSkillHarness:
    def __init__(self, catalog: SkillCatalog | None = None, dry_run: bool = False):
        self.catalog = catalog or SkillCatalog()
        self.dry_run = dry_run
        self.loaded_files: list[str] = []

    def run(self, request: str) -> SkillRunResult:
        """完整流程：catalog 路由 → 加载 SKILL.md → 按需加载 references → 规划 → 可选执行。"""
        self.loaded_files = []
        skill = self.match_skill(request)
        if not skill:
            return SkillRunResult(request=request, matched_skill=None, error="没有匹配到可用 skill")

        result = SkillRunResult(request=request, matched_skill=skill.name)
        try:
            skill_body = self.load_skill_body(skill)
            references = self.load_references_for_request(skill, request)
            result.assistant_plan = self.make_plan(request, skill, skill_body, references)
            commands, artifacts = self.maybe_execute_builtin(request, skill)
            result.executed_commands = commands
            result.artifacts = artifacts
            result.loaded_files = self.loaded_files
            return result
        except Exception as e:
            result.error = str(e)
            result.loaded_files = self.loaded_files
            return result

    # ---------- 阶段1：只用轻量 catalog 匹配 ----------

    def match_skill(self, request: str) -> SkillMeta | None:
        skills = self.catalog.scan()
        if not skills:
            return None

        # 优先让 LLM 只根据 name/description 路由。
        try:
            prompt = (
                "你是 Skills Harness 的路由器。只根据 skill 名称和 description 判断是否匹配用户请求。\n"
                "如果没有合适 skill，返回 {\"skill\": null, \"reason\": \"...\"}。\n"
                "如果有，返回 {\"skill\": \"技能名\", \"reason\": \"...\"}。只返回 JSON。\n\n"
                f"用户请求：{request}\n\n可用 skills：\n"
                + "\n".join(f"- {s.name}: {s.description}" for s in skills)
            )
            client, model = get_chat_client()
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            data = parse_json_object(resp.choices[0].message.content)
            skill_name = data.get("skill") if data else None
            if skill_name:
                return next((s for s in skills if s.name == skill_name), None)
        except Exception:
            pass

        # 无 API Key / LLM 失败时兜底，便于课堂演示。
        return self.keyword_match(request, skills)

    @staticmethod
    def keyword_match(request: str, skills: list[SkillMeta]) -> SkillMeta | None:
        req = request.lower()
        for skill in skills:
            name_desc = f"{skill.name} {skill.description}".lower()
            if any(k in req for k in ["闪卡", "flash card", "单词卡"]):
                if "flash" in name_desc or "闪卡" in name_desc:
                    return skill
            if any(k in req for k in ["图", "diagram", "流程", "架构", "时序", "sequence", "flowchart"]):
                if "diagram" in name_desc or "图" in name_desc:
                    return skill
        return None

    # ---------- 阶段2：命中后才读完整 SKILL.md ----------

    def load_skill_body(self, skill: SkillMeta) -> str:
        self.loaded_files.append(str(skill.path))
        return read_limited(skill.path)

    # ---------- 阶段3：按需读取 references ----------

    def load_references_for_request(self, skill: SkillMeta, request: str) -> list[tuple[Path, str]]:
        base = skill.path.parent
        lower = request.lower()
        wanted: list[Path] = []

        if any(k in lower for k in ["时序", "sequence"]):
            wanted.append(base / "references" / "sequence.md")
        if any(k in lower for k in ["架构", "architecture"]):
            wanted.append(base / "references" / "architecture.md")
        if any(k in lower for k in ["流程", "flowchart", "flow chart"]):
            wanted.append(base / "references" / "flowchart.md")
        if any(k in lower for k in ["结构", "类图", "er", "structural"]):
            wanted.append(base / "references" / "structural.md")

        loaded: list[tuple[Path, str]] = []
        for path in dict.fromkeys(wanted):
            if path.exists():
                self.loaded_files.append(str(path))
                loaded.append((path, read_limited(path)))
        return loaded

    # ---------- 阶段4：规划 ----------

    def make_plan(self, request: str, skill: SkillMeta, skill_body: str, references: list[tuple[Path, str]]) -> str:
        ref_text = "\n\n".join(f"## Reference: {p.name}\n{text}" for p, text in references)
        prompt = (
            "你是一个执行 Skills 的 Harness。根据已加载的 skill 文档，给出可执行步骤。\n"
            "如果需要创建文件，请给出文件路径和内容摘要；如果需要运行脚本，请给出命令。\n"
            "不要假设未加载的文件内容。\n\n"
            f"用户请求：{request}\n\n"
            f"已加载 SKILL.md：\n{skill_body}\n\n{ref_text}"
        )
        try:
            client, model = get_chat_client()
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"已匹配 {skill.name}。LLM 规划跳过：{e}\n已加载文件：" + ", ".join(self.loaded_files)

    # ---------- 阶段5：可选内置执行器 ----------

    def maybe_execute_builtin(self, request: str, skill: SkillMeta) -> tuple[list[str], list[str]]:
        """
        为了让作业能完整演示“执行 scripts”，内置 flash-card 的最小执行逻辑。
        其他 skill 默认只规划不执行。
        """
        if self.dry_run or skill.name != "flash-card":
            return [], []

        word = extract_english_word(request)
        if not word:
            return [], []

        data_dir = skill.path.parent / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        json_path = data_dir / f"{word}.json"
        html_path = PROJECT_ROOT / f"{word}.html"

        data = make_flashcard_data(word)
        json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

        script = skill.path.parent / "scripts" / "make_flashcard.py"
        cmd = [sys.executable, str(script), str(json_path), "-o", str(html_path)]
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
        return [" ".join(cmd)], [str(json_path), str(html_path)]


# ============================================================
# 5. 工具函数
# ============================================================

def read_limited(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if len(text) > MAX_INLINE_FILE_CHARS:
        return text[:MAX_INLINE_FILE_CHARS] + "\n\n[内容过长，已截断]"
    return text


def parse_json_object(text: str) -> dict[str, Any] | None:
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text.strip())
    text = re.sub(r"\n?```$", "", text.strip())
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return None


def extract_english_word(request: str) -> str | None:
    words = re.findall(r"\b[a-zA-Z][a-zA-Z-]{1,30}\b", request)
    stop = {"flash", "card", "html", "make", "word"}
    for word in words:
        if word.lower() not in stop:
            return word.lower()
    return None


def make_flashcard_data(word: str) -> dict:
    """无 LLM 时的演示数据；实际提交重点是 Harness 渐进式加载。"""
    return {
        "word": word,
        "phonetic": "/请补充音标/",
        "pos": "n./v./adj.",
        "definition": "课堂 Harness 演示生成的占位释义；实际使用时可由 LLM 生成更准确的数据。",
        "examples": [
            {"en": f"This is an example sentence for {word}.", "zh": f"这是一个包含 {word} 的示例句。"},
            {"en": f"Try to use {word} in your own sentence.", "zh": f"试着在自己的句子里使用 {word}。"},
            {"en": f"The word {word} can be learned with context.", "zh": f"单词 {word} 可以结合语境学习。"},
        ],
        "synonyms": ["related", "similar", "near", "connected"],
    }


# ============================================================
# 6. CLI
# ============================================================

def list_skills():
    catalog = SkillCatalog()
    skills = catalog.scan()
    print("Skill Catalog（只读取 frontmatter）：")
    for s in skills:
        print(f"- {s.name}: {s.description}")
        print(f"  path: {s.path}")


def main():
    parser = argparse.ArgumentParser(description="作业13：渐进式加载执行 Skills 的 Harness")
    parser.add_argument("request", nargs="*", help="用户请求")
    parser.add_argument("--dry-run", action="store_true", help="只展示匹配和加载，不执行脚本")
    parser.add_argument("--list", action="store_true", help="列出 skill catalog")
    args = parser.parse_args()

    if args.list:
        list_skills()
        return

    if not args.request:
        parser.error("请提供用户请求，或使用 --list")

    request = " ".join(args.request)
    harness = ProgressiveSkillHarness(dry_run=args.dry_run)
    result = harness.run(request)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
