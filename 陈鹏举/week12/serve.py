import uuid
from fastapi import HTTPException

# 会话存储（简单内存字典，生产环境建议使用 Redis）
sessions = {}
session_locks = {}  # 每个会话独立锁，防止并发冲突

@app.post("/query/manual")
async def query_manual(req: QueryRequest, session_id: str = None):
    if session_id is None:
        session_id = str(uuid.uuid4())
    # 获取或创建会话历史
    async with get_session_lock(session_id):
        history = sessions.get(session_id)
        if history is None:
            from react_manual import SYSTEM_PROMPT
            history = [{"role": "system", "content": SYSTEM_PROMPT}]
            sessions[session_id] = history

    # 流式响应，在 start 事件中返回 session_id
    async def event_generator():
        yield _sse({"type": "start", "session_id": session_id, "question": req.question, "mode": "manual"})
        # 使用独立的锁，确保历史更新原子性
        async with get_session_lock(session_id):
            # 调用 run，传入 history（可变列表）
            from react_manual import run
            # 由于 run 是同步生成器，在异步中执行需在线程池中运行
            loop = asyncio.get_event_loop()
            gen = run(req.question, max_steps=req.max_steps, history=history)
            # 逐个获取 step，并通过 SSE 发送
            for step_data in gen:
                yield _sse(step_data)
            # 生成器结束后，history 已被更新，无需额外操作
        yield _sse({"type": "done"})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

# 类似地实现 /query/fc，只需替换 import 和 SYSTEM_PROMPT
