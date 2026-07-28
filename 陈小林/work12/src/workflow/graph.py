from langgraph.graph import END, StateGraph
from langchain_openai import ChatOpenAI

from src.rag.md_init import HybridRetriever
from src.workflow.state import GraphState
from src.workflow.nodes import (
    analyze,
    rewrite,
    retrieve,
    grade,
    generate,
    verify,
)

def build_rag_graph(llm: ChatOpenAI, retriever:HybridRetriever):
    workflow = StateGraph(GraphState)

    workflow.add_node("analyze", lambda s: analyze(s, llm))
    workflow.add_node("rewrite", lambda s: rewrite(s, llm))
    workflow.add_node("retrieve", lambda s : retrieve(s, retriever))
    workflow.add_node("grade", grade)
    workflow.add_node("generate", lambda s: generate(s, llm))
    workflow.add_node("verify", lambda s: verify(s, llm))

    workflow.set_entry_point("analyze")

    # analyze → rewrite(需要检索) / generate(闲聊直接回)
    workflow.add_conditional_edges(
        "analyze",
        lambda s: "rewrite" if s["needs_retrieval"] else "generate",
    )

    # rewrite → retrieve → grade
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("retrieve", "grade")

    # grade → rewrite(无相关文档，重写重试) / generate
    workflow.add_conditional_edges(
        "grade",
        lambda s: "rewrite" if s["needs_rewrite"] else "generate",
    )

    # generate → verify
    workflow.add_edge("generate", "verify")

    # verify → retrieve(幻觉，重新检索) / END
    workflow.add_conditional_edges(
        "verify",
        lambda s: "retrieve" if s["has_hallucination"] else END,
    )

    return workflow.compile()
