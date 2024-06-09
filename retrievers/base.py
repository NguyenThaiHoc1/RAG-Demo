class BaseRetriever(object):

    def __init__(self, vector_database):
        self.vb = vector_database

    def get_relevant_documents(self):
        pass
