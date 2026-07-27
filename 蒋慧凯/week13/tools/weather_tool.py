"""
weather_tool.py —— Mock 天气查询工具。

教学重点：
  1. 模拟外部 API 调用，避免作业依赖真实天气服务。
  2. 返回固定格式的字符串，便于 Skill 组织语言。
"""

import random


# 预设几个城市的天气，让演示输出可预期
_WEATHER_DB = {
    "北京": "晴，15~28°C，空气质量良",
    "上海": "多云，18~25°C，微风",
    "深圳": "雷阵雨，22~30°C，湿度较高",
    "杭州": "小雨，17~24°C，体感舒适",
}


def get_weather(city: str) -> str:
    """
    查询指定城市的天气。

    参数:
        city: 城市名，例如 "北京"。

    返回:
        天气描述字符串。
    """
    if not city:
        return "错误：未提供城市名"

    city = city.strip()
    if city in _WEATHER_DB:
        return f"{city}：{_WEATHER_DB[city]}"

    # 未收录城市返回随机但稳定的 mock 结果
    random.seed(city)
    conditions = ["晴", "多云", "阴", "小雨", "雷阵雨"]
    condition = random.choice(conditions)
    low = random.randint(10, 20)
    high = low + random.randint(5, 15)
    return f"{city}：{condition}，{low}~{high}°C"
