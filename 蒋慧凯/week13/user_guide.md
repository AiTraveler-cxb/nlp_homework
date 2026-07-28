# Week 13 作业：渐进式 Skill Harness

## 项目简介

本项目实现了一套最小可运行的 **渐进式 Skill Harness**，用于演示 Agent 系统中"常驻索引 + 按需加载 + 执行后释放"的 Skill 生命周期。

核心能力：

- 根据用户输入中的关键词触发对应 Skill。
- 仅在被触发时加载完整 Skill 定义，降低 Context 占用。
- 解析 Skill 中的步骤，调用对应工具函数完成具体任务。
- 支持渐进式与全量加载两种模式，可量化对比 Context 占用差异。

## 目录结构

```
week13harness和skills/homework/
├── user_guide.md             # 本文件
├── requirements.txt          # 依赖说明
├── run_demo.py              # 演示入口
├── harness.py               # Harness 核心实现
├── skills/                  # Skill 定义目录
│   ├── index.md             # 常驻索引
│   ├── weather.md           # 天气查询 Skill
│   ├── calculator.md        # 计算器 Skill
│   └── weekly_report.md     # 周报生成 Skill
├── tools/                   # 工具函数目录
│   ├── __init__.py
│   ├── weather_tool.py
│   ├── calculator_tool.py
│   └── file_tool.py
└── outputs/                 # 运行输出
    ├── run_demo_log.txt
    └── comparison_table.md
```

## 环境要求

- Python >= 3.10
- 仅使用 Python 标准库，无需安装第三方依赖
- 如需使用 PyYAML 解析 frontmatter，可安装：

```bash
pip install pyyaml
```

## 输入输出与运行方式

### 运行演示

```bash
cd week13harness和skills/homework
python run_demo.py
```

### 演示内容

脚本会依次处理以下 3 个查询：

1. `北京今天天气怎么样？` —— 触发 weather Skill，调用 mock 天气工具。
2. `帮我算一下 (15 + 23) * 4 等于多少` —— 触发 calculator Skill，调用安全计算器。
3. `帮我写一份周报` —— 触发 weekly_report Skill，读取任务列表并生成周报文件。

每个查询会展示：

- 命中的 Skill 名称
- 执行步骤与工具返回结果
- 最终回答
- 当前 Context 的 token 占用估算

### 输出文件

- `outputs/run_demo_log.txt`：完整运行日志
- `outputs/comparison_table.md`：渐进式 vs 全量加载 Context 占用对比表
- `data/weekly_report_2026.md`：周报生成 Skill 保存的周报文件

### 新增 Skill

1. 在 `skills/` 目录下新建 Markdown 文件。
2. 文件顶部写入 frontmatter：

```markdown
---
name: skill_name
description: 一句话描述
trigger: 触发词 | 别名
---
```

3. 在 `## Steps` 区域用 `调用 xxx_tool.func(args)` 格式描述工具调用。
4. 在 `skills/index.md` 中添加一行摘要。
