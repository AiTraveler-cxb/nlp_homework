"""
langchain工具相关功能测试
"""
import os
from typing import Annotated

#模型加载
import dotenv

from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain_core.messages import HumanMessage
from langchain_core.tools import InjectedToolCallId
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import InjectedState
from langgraph.types import interrupt

from src.lc.middleware import a1,a2,a3,a4,a5
from src.api.output import WeatherInfo

dotenv.load_dotenv()
model = ChatOpenAI(
            model=os.getenv("model"),
            temperature= 0.7,
            base_url=os.getenv("model_url"),
            api_key=os.getenv("model_appkey"))


from langchain.tools import tool

@dynamic_prompt
def personalized_prompt(request:ModelRequest):
    system_prompt = "你是一个有用的AI助手，你能使用工具回答用户问题"
    mgs = request.messages
    if len(mgs) < 2:
        return system_prompt + ", 对话刚开始，必须要先开一个小玩笑"

    return system_prompt

@tool(return_direct=True)
def get_book_info(book:str,
                  call_id:Annotated[str, InjectedToolCallId],
                  state:Annotated[dict[str,any], InjectedState]) -> str:
    """根据课程名称，获取课程信息，包含课程价格"""
    return f"课程{book}, 30元/时"

@tool
def get_weather_info(city:str,
                  call_id:Annotated[str, InjectedToolCallId],
                  state:Annotated[dict[str,any], InjectedState]) -> str:
    """根据城市名称，获取城市天气信息"""
    # 暂停并等待审批
    approval = interrupt({
        "message": f"确认查询天气吗《{city}》？此操作不可撤销。"
    })
    if approval.get("confirmed"):
        if city == '厦门':
            return '不支持厦门的天气查询'
        return f"城市{city}, 小雨转中雨,50~70度"
    return "查询取消"

def main_query_book():
    agent = create_agent(
        model=model,
        tools=[get_book_info],
        debug=True,
        middleware=[personalized_prompt, a1, a2, a3, a5])

    inputs = {"messages": [HumanMessage(content="语文课价格怎么样？")]}
    res = agent.invoke(inputs)
    for msg in res["messages"]:
        print(msg)

def get_agent():
    checkpointer = MemorySaver()  # 开发测试用
    agent = create_agent(
        model=model,
        tools=[get_book_info,get_weather_info],
        debug=True,
        checkpointer=checkpointer,
        middleware=[personalized_prompt, a1, a2, a3, a5])
    return agent

def main():
    main_query_book()

if __name__ == "__main__":
    main()