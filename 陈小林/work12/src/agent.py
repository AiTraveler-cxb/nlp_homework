import time

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field
from src.memory.base import ConversationMemory, RedisMemory
from src.llm.base import LLMFactory
from src.models.entities import LLMConfig
from src.config_loader import load_config
from src.rag.vector_store import vector_store
from src.workflow.graph import build_rag_graph
from src.rag.md_init import KnowledgeBaseInitializer
from pathlib import Path
ROOT = Path(__file__).parent
class Agent(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    memories: dict[str, ConversationMemory] = Field(default_factory=dict)
    llm_config: LLMConfig = None
    llm: ChatOpenAI = None
    rag_graph: StateGraph = None

    def model_post_init(self, _context):
        kb = KnowledgeBaseInitializer(ROOT, ignore_vec_init = True)
        kb.load_metadata(ROOT / 'kb_metadata')
        self.llm = LLMFactory.create_llm(self.llm_config)
        self.rag_graph = build_rag_graph(llm=self.llm, retriever = kb.build_index())

    def start(self, session_id: str):
        memory = self.memories.get(session_id)
        if memory is None:
            memory = RedisMemory(session_id=session_id, conf=load_config('local.yaml'))
            self.memories[session_id] = memory
        while True:
            user_message = input("你：")
            if user_message == "":
                break

            recent = [
                f"用户：{u}\n助手：{a}"
                for u, a in memory.messages[-3:]  # 最近 3 轮
            ]

            result = self.rag_graph.invoke({
                "question": user_message,
                "rewritten_question": "",
                "keywords": "",
                "documents": [],
                "filtered_docs": [],
                "generation": "",
                "needs_retrieval": False,
                "needs_rewrite": False,
                "has_hallucination": False,
                "retrieval_attempts": 0,
                "session_id": session_id,
                "recent_history": recent,
            })

            response = result["generation"]
            print(f"助手：{response}")
            memory.add_message(user_message, response)

            # 同时写入向量库，供后续 RAG 检索
            # vector_store.add_text(
            #     text=f"用户：{user_message}\n助手：{response}",
            #     metadata={"session_id": session_id, "timestamp": time.time()}
            # )





