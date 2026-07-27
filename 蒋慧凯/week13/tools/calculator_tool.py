"""
calculator_tool.py —— 受限表达式计算器。

教学重点：
  1. 使用 eval 时必须做严格白名单校验，避免任意代码执行。
  2. 只允许数字、括号、四则运算和常用数学函数。
"""

import ast
import operator
import math


# 允许的二元/一元运算符
_ALLOWED_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}

_ALLOWED_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# 允许调用的函数名（白名单）
_ALLOWED_NAMES = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "log": math.log,
    "log10": math.log10,
    "abs": abs,
    "round": round,
    "max": max,
    "min": min,
}


def _eval_node(node):
    """递归求值 AST 节点，遇到不允许的类型立即抛异常。"""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"不支持的常量类型: {type(node.value)}")

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_BIN_OPS:
            raise ValueError(f"不支持的运算符: {op_type.__name__}")
        return _ALLOWED_BIN_OPS[op_type](_eval_node(node.left), _eval_node(node.right))

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_UNARY_OPS:
            raise ValueError(f"不支持的一元运算符: {op_type.__name__}")
        return _ALLOWED_UNARY_OPS[op_type](_eval_node(node.operand))

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("只允许调用简单函数名")
        func_name = node.func.id
        if func_name not in _ALLOWED_NAMES:
            raise ValueError(f"不允许调用的函数: {func_name}")
        args = [_eval_node(arg) for arg in node.args]
        return _ALLOWED_NAMES[func_name](*args)

    if isinstance(node, ast.Name):
        if node.id not in _ALLOWED_NAMES:
            raise ValueError(f"不允许使用的名称: {node.id}")
        return _ALLOWED_NAMES[node.id]

    raise ValueError(f"不支持的表达式节点: {type(node).__name__}")


def calculate(expr: str) -> str:
    """
    安全计算数学表达式。

    参数:
        expr: 数学表达式字符串，例如 "(10 + 20) * 2" 或 "sqrt(16)".

    返回:
        计算结果字符串。
    """
    if not expr or not expr.strip():
        return "错误：表达式为空"

    try:
        tree = ast.parse(expr.strip(), mode="eval")
        result = _eval_node(tree)
        return f"{expr} = {result}"
    except Exception as e:
        return f"计算错误：{e}"
