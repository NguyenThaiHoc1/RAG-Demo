from langchain.retrievers import MultiQueryRetriever

from retrievers.base import BaseRetriever
from typing import List

from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field


from langchain_community.chat_models import ChatOllama


class LineList(BaseModel):
    # "lines" is the key (attribute name) of the parsed output
    lines: List[str] = Field(description="Lines of text")


class LineListOutputParser(PydanticOutputParser):
    def __init__(self) -> None:
        super().__init__(pydantic_object=LineList)

    def parse(self, text: str) -> LineList:
        lines = text.strip().split("\n")
        return LineList(lines=lines)


class MultiRetriever(BaseRetriever):

    def __init__(self, vector_database, llm_model):
        self.model = llm_model
        super().__init__(vector_database=vector_database)

    def get_relevant_documents(self):
        output_parser = LineListOutputParser()

        QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""You are the AI language modeling assistant. Your task is to create five
             different versions of a given user query to retrieve related documents from a vector
             database. By creating multiple perspectives on the user's question, your goal is to help
             users overcome some of the limitations of distance-based similarity searches.
             Provide these alternative questions separated by newlines.
             Please Vietnamese responses.
             Original question: {question}""",
        )
        llm = self.model

        retriever = MultiQueryRetriever.from_llm(
            self.vb.as_retriever(search_type="similarity", search_kwargs={"k": 10}), llm, prompt=QUERY_PROMPT
        )  # "lines" is the key (attribute name) of the parsed output

        return retriever
