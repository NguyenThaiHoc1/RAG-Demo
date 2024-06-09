from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List


class PDFDocumentReProcessing(object):

    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self._text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                             chunk_overlap=chunk_overlap,
                                                             add_start_index=True)

    def processing(self, file_path, using_splitter=True) -> List[Document]:
        loader = PyPDFLoader(file_path)
        document_list = loader.load_and_split()

        if using_splitter:
            document_list = self._text_splitter.split_documents(document_list)

        return document_list
