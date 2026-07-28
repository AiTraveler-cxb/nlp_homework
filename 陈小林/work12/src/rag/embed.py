from langchain_huggingface import HuggingFaceEmbeddings

_default_embedding = None


def get_embedding(model_name: str = "BAAI/bge-small-zh"):
    global _default_embedding
    if _default_embedding is None:
        _default_embedding = HuggingFaceEmbeddings(model_name=model_name)
    return _default_embedding

if __name__== '__main__':
    get_embedding()