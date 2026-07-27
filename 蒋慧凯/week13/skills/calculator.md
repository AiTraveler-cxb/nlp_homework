---
name: calculator
description: 执行数学表达式计算
trigger: 计算 | 算一下 | calculator | 等于
---

# 计算器 Skill

## 适用场景
用户需要计算一个数学表达式的结果。

## Steps
1. 从用户输入中提取数学表达式
2. 调用 calculator_tool.calculate(expr)
3. 直接返回计算结果

## 输出格式
返回 "表达式 = 结果"，例如 "(10 + 20) * 2 = 60"。

## 自检
- [ ] 表达式已提取
- [ ] 计算结果无报错
