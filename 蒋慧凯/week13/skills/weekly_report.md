---
name: weekly_report
description: 根据本周任务列表生成周报
trigger: 周报 | 写周报 | weekly report
---

# 周报生成 Skill

## 适用场景
用户希望根据本周完成的任务列表生成一份周报。

## Steps
1. 调用 file_tool.read_file("weekly_tasks.md") 获取本周任务列表
2. 汇总已完成和未完成任务
3. 生成包含本周进展、下周计划、存在风险的周报
4. 调用 file_tool.write_file("weekly_report_2026.md", content) 保存周报

## 输出格式
先输出周报内容摘要，再显示保存结果。

## 内嵌知识
- 周报结构：本周进展 / 下周计划 / 风险与建议
- 任务状态：已完成任务用 [x]，未完成任务用 [ ]

## 自检
- [ ] 任务文件读取成功
- [ ] 周报内容包含三大模块
- [ ] 文件写入成功
