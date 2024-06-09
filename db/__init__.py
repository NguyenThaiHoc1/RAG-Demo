from langchain_community.vectorstores import Chroma

from embedding_model import (
    model_embedding_huggingface
)

# CHROMA
langchain_chroma = Chroma(
    embedding_function=model_embedding_huggingface,
    persist_directory='db/chroma_db'
)
