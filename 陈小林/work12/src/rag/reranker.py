from concurrent.futures import ThreadPoolExecutor, TimeoutError

from sentence_transformers import CrossEncoder

from langchain_core.documents import Document

_default_reranker = None
_pool = ThreadPoolExecutor(max_workers=1)
_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
_RERANK_TIMEOUT = 30


def _get_reranker():
    global _default_reranker
    if _default_reranker is None:
        _default_reranker = CrossEncoder(_MODEL_NAME)
        # 预热：跑一次 dummy 推理，避免第一次请求卡住
        _default_reranker.predict([["预热", "测试"]])
    return _default_reranker


def rerank(query: str, docs: list[Document], threshold: float = 0.4) -> list[Document]:
    if not docs:
        return []

    reranker = _get_reranker()
    pairs = [[query, doc.get('text')] for doc in docs]

    future = _pool.submit(reranker.predict, pairs)
    try:
        scores = future.result(timeout=_RERANK_TIMEOUT)
    except TimeoutError:
        print("CrossEncoder 超时，降级为向量分数过滤")
        return [d for d in docs if d.metadata.get("retriever_score", 1) < 0.7]

    filtered = []
    for doc, score in zip(docs, scores):
        if score > threshold:
            doc.metadata["rerank_score"] = float(score)
            filtered.append(doc)
    if len(filtered) == 0:
        return docs
    return filtered
