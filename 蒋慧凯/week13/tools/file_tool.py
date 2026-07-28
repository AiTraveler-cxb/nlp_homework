"""
file_tool.py —— 受限文件读写工具。

教学重点：
  1. 限制只能读取项目内 data/ 目录下的文件，防止越权访问。
  2. 写入操作返回确认信息，方便 Skill 自检。
"""

import os
from pathlib import Path


def _safe_path(base_dir: Path, filename: str) -> Path:
    """
    解析文件名并确保结果位于 base_dir 之下，防止目录穿越。
    """
    # 去掉前导路径分隔符和 .. 序列
    cleaned = filename.strip("/\\")
    if ".." in cleaned:
        raise ValueError("文件名不允许包含 ..")

    target = (base_dir / cleaned).resolve()
    base = base_dir.resolve()

    # 确保 target 位于 base 之下
    if base not in target.parents and target != base:
        raise ValueError(f"访问越界: {filename}")

    return target


def read_file(filename: str, base_dir: str = None) -> str:
    """
    读取项目 data/ 目录下的文本文件。

    参数:
        filename: 相对于 data/ 目录的文件名。
        base_dir: 可选，自定义基础目录，默认使用 tools/data。

    返回:
        文件内容字符串。
    """
    if base_dir is None:
        base_dir = Path(__file__).parent.parent / "data"
    else:
        base_dir = Path(base_dir)

    try:
        target = _safe_path(base_dir, filename)
        if not target.exists():
            return f"文件不存在: {filename}"
        with open(target, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"读取失败: {e}"


def write_file(filename: str, content: str, base_dir: str = None) -> str:
    """
    向项目 data/ 目录写入文本文件。

    参数:
        filename: 相对于 data/ 目录的文件名。
        content: 要写入的内容。
        base_dir: 可选，自定义基础目录。

    返回:
        写入结果确认信息。
    """
    if base_dir is None:
        base_dir = Path(__file__).parent.parent / "data"
    else:
        base_dir = Path(base_dir)

    try:
        target = _safe_path(base_dir, filename)
        base_dir.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            f.write(content)
        return f"已写入: {filename}"
    except Exception as e:
        return f"写入失败: {e}"
