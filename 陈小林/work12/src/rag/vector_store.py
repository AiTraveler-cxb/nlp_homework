from chromadb.config import Settings
from langchain_chroma import Chroma

from src.rag.embed import get_embedding


class VectorStore:
    def __init__(self, collection_name="conversations", persist_directory=r"/Users/cxl/voice-agent/src/chroma_data"):
        self._collection_name = collection_name
        self._persist_directory = persist_directory
        self._store = None

    @property
    def store(self):
        if self._store is None:
            self._store = Chroma(
                collection_name=self._collection_name,
                embedding_function=get_embedding(),
                persist_directory=self._persist_directory,
                client_settings=Settings(anonymized_telemetry=False),
            )
        return self._store

    def add_text(self, text: str, metadata: dict):
        self.store.add_texts(texts=[text], metadatas=[metadata])

    def search(self, query: str, k: int = 10) -> list:
        results = self.store.similarity_search_with_score(query, k=k)
        docs = []
        for doc, score in results:
            doc.metadata["retriever_score"] = float(score)
            docs.append(doc)
        return docs


vector_store = VectorStore()
