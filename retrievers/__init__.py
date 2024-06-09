from retrievers.normal import NormalRetriever
from retrievers.multi_query_retrievers import MultiRetriever

from db import (
    langchain_chroma
)

from chatting_model import (
    model_chat_ollama
)

retrievers_normal = NormalRetriever(vector_database=langchain_chroma)
retrievers_multiquery = MultiRetriever(vector_database=langchain_chroma, llm_model=model_chat_ollama.model)
