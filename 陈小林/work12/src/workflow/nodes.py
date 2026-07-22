from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.rag.md_init import HybridRetriever
from src.rag.reranker import rerank
from src.rag.vector_store import vector_store
from src.workflow.state import GraphState

MAX_RETRIEVAL = 2


class _AnalysisResult(BaseModel):
    needs_retrieval: bool = Field(description="是否需要检索历史对话")
    keywords: str = Field(description="提取的搜索关键词")


class _VerificationResult(BaseModel):
    has_hallucination: bool = Field(description="是否发现幻觉")
    reason: str = Field(description="判断理由")


_analysis_parser = PydanticOutputParser(pydantic_object=_AnalysisResult)
_verify_parser = PydanticOutputParser(pydantic_object=_VerificationResult)


def analyze(state: GraphState, llm: ChatOpenAI) -> dict:
    messages = [
        SystemMessage(
            "你是一个对话分析助手。判断用户的问题是普通闲聊，还是需要参考历史对话才能准确回答。\n\n"
            "需要检索历史的情况：问题涉及仓库管理，仓库内流程操作的问题。\n"
            "不需要检索的情况：问候、通用知识、闲聊、天气、时间等独立问题。\n\n"
            "同时提取搜索关键词（2-6个词），用于后续向量检索。\n"
            f"{_analysis_parser.get_format_instructions()}"
        ),
        HumanMessage(f"用户问题：{state['question']}"),
    ]

    response = llm.invoke(messages)
    result = _analysis_parser.invoke(response)

    return {
        "needs_retrieval": result.needs_retrieval,
        "keywords": result.keywords,
    }


def rewrite(state: GraphState, llm: ChatOpenAI) -> dict:
    messages = [
        SystemMessage(
            "将用户的问题改写成更适合向量检索的形式。\n"
            "要求：提取关键实体、属性、术语，补充完整的主谓宾结构，省略口语化表达。\n"
            "直接输出改写后的文本，不要额外解释。"
        ),
        HumanMessage(f"用户问题：{state['question']}"),
    ]

    response = llm.invoke(messages)
    return {"rewritten_question": response.content.strip()}


def retrieve(state: GraphState, retriever:HybridRetriever) -> dict:
    query = state.get("rewritten_question") or state["question"]
    #这里使用 向量+bm25混合检索
    docs = retriever.search(query, 3)

    return {"documents": docs}


def grade(state: GraphState, _=None) -> dict:
    if not state["documents"]:
        return {
            "filtered_docs": [],
            "needs_rewrite": False,
            "retrieval_attempts": state["retrieval_attempts"],
        }

    filtered_docs = rerank(state["question"], state["documents"], threshold=0.4)

    if len(filtered_docs) == 0 and state["retrieval_attempts"] < MAX_RETRIEVAL:
        return {
            "filtered_docs": filtered_docs,
            "needs_rewrite": True,
            "retrieval_attempts": state["retrieval_attempts"] + 1,
        }

    return {
        "filtered_docs": filtered_docs,
        "needs_rewrite": False,
        "retrieval_attempts": state["retrieval_attempts"],
    }


def generate(state: GraphState, llm: ChatOpenAI) -> dict:
    context_parts = []

    if state["recent_history"]:
        recent = "\n".join(state["recent_history"][-6:])  # 最近 3 轮
        context_parts.append(f"[最近对话]\n{recent}")

    if state["filtered_docs"]:
        for i, doc in enumerate(state["filtered_docs"]):
            context_parts.append(f"[参考文档 {i + 1}]\n{doc['text']}")

    if context_parts:
        context = "\n---\n".join(context_parts)
        system_prompt = (
            "你是一个AI助手。请基于以下相关历史对话和用户当前问题来回答。\n"
            "如果历史信息足以回答，请结合历史和你的知识给出准确回答。\n"
            "如果历史信息不足以回答，如实告知用户。"
        )
        user_content = f"相关文档：\n{context}\n\n用户问题：{state['question']}"
    else:
        system_prompt = "你是一个有用的AI助手。请直接回答用户的问题。"
        user_content = state["question"]

    messages = [
        SystemMessage(system_prompt),
        HumanMessage(user_content),
    ]

    response = llm.invoke(messages)
    return {"generation": response.content}


def verify(state: GraphState, llm: ChatOpenAI) -> dict:
    if state["retrieval_attempts"] >= MAX_RETRIEVAL:
        return {"has_hallucination": False}

    if not state["filtered_docs"]:
        return {"has_hallucination": False}

    context = "\n---\n".join(doc['text'] for doc in state["filtered_docs"])

    messages = [
        SystemMessage(
            "检查助手的回答是否基于提供的参考文档。\n"
            "如果回答中的关键信息（数据、配置、结论、具体建议）在参考文档中找不到依据，标记为幻觉。\n"
            "如果是通用知识或合理推断，不算幻觉。\n"
            f"{_verify_parser.get_format_instructions()}"
        ),
        HumanMessage(
            f"参考文档：\n{context}\n\n"
            f"助手回答：{state['generation']}"
        ),
    ]

    response = llm.invoke(messages)
    result = _verify_parser.invoke(response)

    return {
        "has_hallucination": result.has_hallucination,
        "retrieval_attempts": state["retrieval_attempts"] + (1 if result.has_hallucination else 0),
    }
