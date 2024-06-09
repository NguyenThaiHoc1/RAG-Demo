from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings


class HFEmbeddingModel:

    def __init__(self, model_name, model_kwargs, encode_kwargs):
        self.model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.embed_documents(texts)

    def embed_query(self, query: str) -> List[float]:
        return self.model.embed_query(query)
