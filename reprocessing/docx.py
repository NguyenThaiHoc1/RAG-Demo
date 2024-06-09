from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List


class DocxDocumentReProcessing(object):

    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self._text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def processing(self, file_path, using_splitter=True) -> List[Document]:
        loader = Docx2txtLoader(file_path)
        document_list = loader.load_and_split()

        if using_splitter:
            document_list = self._text_splitter.split_documents(document_list)

        return document_list
