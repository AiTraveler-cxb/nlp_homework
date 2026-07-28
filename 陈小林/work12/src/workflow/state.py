from typing import List, TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict):
    question: str
    rewritten_question: str
    keywords: str
    documents: List[Document]
    filtered_docs: List[Document]
    generation: str
    needs_retrieval: bool
    needs_rewrite: bool
    has_hallucination: bool
    retrieval_attempts: int
    session_id: str
    recent_history: List[str]
