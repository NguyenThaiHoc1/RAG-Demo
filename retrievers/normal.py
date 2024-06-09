from retrievers.base import BaseRetriever


class NormalRetriever(BaseRetriever):

    def __init__(self, vector_database):
        super().__init__(vector_database=vector_database)

    def get_relevant_documents(self):
        retriever = self.vb.as_retriever(search_type="similarity", search_kwargs={"k": 10})
        return retriever
