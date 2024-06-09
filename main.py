
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma

from embedding_model import (
    model_embedding_huggingface
)

from chatting_model import (
    model_chat_ollama
)

urls = [
    "https://docs-aiservice.fujinet.net"
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs_list)

db = Chroma.from_documents(
    splits,
    embedding=model_embedding_huggingface,
    persist_directory='./vectordatabase/chroma'
)

chat = model_chat_ollama.do_activate(
    query="How many services that provided by Fujinet ?",
    retriever=db.as_retriever(search_type="similarity", k=2)
)

print(chat)

