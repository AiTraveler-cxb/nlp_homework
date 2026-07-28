import json
import time
import uuid

import uvicorn
from fastapi import FastAPI
from langchain_core.messages import HumanMessage
from langgraph.errors import GraphInterrupt
from langgraph.types import Command
from starlette.responses import StreamingResponse

from src.api.input import Msg
from src.lc.tools import get_agent

app = FastAPI()
interrupt_states = {}
agent = get_agent()

@app.get("/")
async def hello_world():
    return {"message": "Hello World"}

@app.get("/chat")
async def chat(message: str, session_id: str = None):
    confirmed = False
    if session_id:
        confirmed = True
    session_id = session_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}

    async def do_chat():

        if not confirmed:
            async for msg in new_chat(message, session_id, config):
                yield msg
        else :
            async for msg_chunk in agent.astream_events(
                                                Command(resume={"confirmed": confirmed}),
                                                config=config):

                if msg_chunk['event'] == 'on_chat_model_stream':
                    chunk = msg_chunk['data']['chunk']
                    if chunk.content:
                        yield chunk.content
                if msg_chunk['event'] == 'on_tool_error':
                    err = msg_chunk["data"]["error"]
                    if isinstance(err, GraphInterrupt):
                        interrupt_msg = msg_chunk["data"]["error"].args[0][0].value['message']

                        # 保存当前状态（包含中断信息和恢复方式）
                        interrupt_states[session_id] = {
                            "config": config,
                            # 不需要保存 state，checkpoint 已经保存了
                            "interrupt_data": interrupt_msg,
                            "timestamp": time.time()
                        }

                        # 返回中断事件给前端
                        yield f"data: {json.dumps({
                            'type': 'interrupt',
                            'thread_id': session_id,
                            'data': interrupt_msg
                        })}\n\n"
                        return

    return StreamingResponse(do_chat(), media_type="text/event-stream")

async def new_chat(message: str, session_id: str = None, config: dict = None):
    async for msg_chunk in agent.astream_events(
            {"messages": [HumanMessage(content=message)]}, config=config):
        if msg_chunk['event'] == 'on_chat_model_stream':
            chunk = msg_chunk['data']['chunk']
            if chunk.content:
                yield chunk.content
        if msg_chunk['event'] == 'on_tool_error':
            err = msg_chunk["data"]["error"]
            if isinstance(err, GraphInterrupt):
                interrupt_msg = msg_chunk["data"]["error"].args[0][0].value['message']

                # 保存当前状态（包含中断信息和恢复方式）
                interrupt_states[session_id] = {
                    # 不需要保存 state，checkpoint 已经保存了
                    "interrupt_data": interrupt_msg,
                    "timestamp": time.time()
                }

                # 返回中断事件给前端
                yield f"data: {json.dumps({
                    'type': 'interrupt',
                    'thread_id': session_id,
                    'data': interrupt_msg
                })}\n\n"
                return

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
