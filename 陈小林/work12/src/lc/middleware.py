from langchain.agents import AgentState
from langchain.agents.middleware import before_agent, before_model, after_model, wrap_tool_call, after_agent
from langchain_core.messages import HumanMessage


@before_agent
def a1(state: AgentState, runtime):
    print(">>> [before_agent] Agent 开始 <<<")
    return None

@before_model
def a2(state: AgentState, runtime):
    print(">>> [before_model] Agent 开始 <<<")
    return None

@after_model
def a3(state: AgentState, runtime):
    print(">>> [after_model] Agent 开始 <<<")
    return None

@wrap_tool_call
def a4(state: AgentState, runtime):
    print(">>> [wrap_tool_call] Agent 开始 <<<")
    return None

@after_agent()
def a5(state: AgentState, runtime):
    print(">>> [after_agent] Agent 开始 <<<")
    return None