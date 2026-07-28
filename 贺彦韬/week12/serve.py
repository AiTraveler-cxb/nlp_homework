"""
FastAPI HTTP 服务，提供流式 SSE 接口给 Web UI

接口：
  POST /query/manual  - 手写版 ReAct，流式返回每步
  POST /query/fc      - Function Calling 版，流式返回每步
  GET  /health        - 健康检查

使用方式：
  uvicorn serve:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import json
import logging
import asyncio
from threading import Lock
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── 预加载 FAISS（启动时执行一次）────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("预加载 FAISS 索引和 Embedding 模型...")
    from tools import _load_rag
    await asyncio.to_thread(_load_rag)
    logger.info("预加载完成，服务就绪")
    yield


app = FastAPI(title="ReAct Financial Agent", lifespan=lifespan)


# ── 全局历史存储（不分用户/会话，全局共用一份记忆） ─────────────────────────
# 注意：存在内存里，服务重启就会清空；且所有请求共用同一份历史。
# manual 和 fc 分开存，因为两者的 system prompt 不一样。
_HISTORY: dict[str, list] = {}
_HISTORY_LOCK = Lock()


def _get_history(mode: str) -> list:
    from react_manual import SYSTEM_PROMPT
    from react_function_calling import FC_SYSTEM_PROMPT

    with _HISTORY_LOCK:
        if mode not in _HISTORY:
            prompt = SYSTEM_PROMPT if mode == "manual" else FC_SYSTEM_PROMPT
            _HISTORY[mode] = [{"role": "system", "content": prompt}]
        return _HISTORY[mode]


# ── 请求/响应模型 ─────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question:  str
    max_steps: int = 10


# ── SSE 流式生成器 ────────────────────────────────────────────────────────────
def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _stream_react(question: str, max_steps: int, mode: str):
    """
    同步生成器（react_run）在独立线程中逐步执行，
    每产出一步通过 asyncio.Queue 传递给异步 SSE 生成器，
    实现真正的边思考边推送。

    每次调用都会取出 mode 对应的全局历史一起传给 react_run，
    这样上一次问答的内容会保留，下一次提问能接着问。
    """
    if mode == "manual":
        from react_manual import run as react_run
    else:
        from react_function_calling import run as react_run

    messages = _get_history(mode)

    queue: asyncio.Queue = asyncio.Queue()
    _SENTINEL = object()

    def _worker():
        try:
            for step_data in react_run(question, max_steps=max_steps, messages=messages):
                queue.put_nowait(step_data)
        finally:
            queue.put_nowait(_SENTINEL)

    yield _sse({"type": "start", "question": question, "mode": mode})

    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _worker)

    while True:
        step_data = await queue.get()
        if step_data is _SENTINEL:
            break
        yield _sse(step_data)

    yield _sse({"type": "done"})


# ── 路由 ──────────────────────────────────────────────────────────────────────
@app.post("/query/manual")
async def query_manual(req: QueryRequest):
    return StreamingResponse(
        _stream_react(req.question, req.max_steps, "manual"),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/query/fc")
async def query_fc(req: QueryRequest):
    return StreamingResponse(
        _stream_react(req.question, req.max_steps, "fc"),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model": os.getenv("AGENT_MODEL", "qwen-max")}


@app.post("/reset")
async def reset(mode: str | None = None):
    """
    清空记忆，重新开始。
    不传 mode：manual 和 fc 的历史都清空。
    传 mode=manual 或 mode=fc：只清空对应那一份。
    """
    with _HISTORY_LOCK:
        if mode is None:
            _HISTORY.clear()
        else:
            _HISTORY.pop(mode, None)
    return {"status": "ok", "reset": mode or "all"}


@app.get("/debug/messages")
async def debug_messages(mode: str = "manual"):
    """
    查看当前 mode（manual 或 fc）的完整 messages 历史，方便调试记忆功能。
    浏览器直接访问 http://localhost:8000/debug/messages?mode=manual 即可查看。
    """
    with _HISTORY_LOCK:
        messages = _HISTORY.get(mode, [])
        return {
            "mode":          mode,
            "message_count": len(messages),
            "messages":      messages,
        }


# ── 托管 index.html ──────────────────────────────────────────────────────────
HTML_PATH = Path(__file__).parent.parent / "index.html"

@app.get("/")
async def root():
    if HTML_PATH.exists():
        return HTMLResponse(HTML_PATH.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>index.html not found</h2>")
