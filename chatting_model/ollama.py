from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts import PromptTemplate

from langchain_community.chat_models import ChatOllama
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


def format_docs(docs):
    str_total = ""
    for doc in docs:
        content = doc.page_content
        str_total += content + "\n" + "--------------------------------------------------------------------------------" + "\n"
    return str_total
    # return "\n\n".join(doc.page_content for doc in docs)


class OllamaChat(object):

    def __init__(self, model_name):
        self.model = ChatOllama(model=model_name, temperature=0, num_ctx=4096 * 4)
        # self.rag_prompt_template = PromptTemplate.from_template(self.rag_prompt)

    def do_activate(self, query, retriever, *args, **kwargs):
        ### Answer question ###
        qa_system_prompt = """
        You are an assistant named Mr. Hoc, specializing in answering User's requests based on provided documents. \
        Mr. Hoc will find answer in the retrieved documents excerpts to respond to User's requests. \
        If you don't know the answer, Mr. Hoc will acknowledge that you don't know. \
        Mr. Hoc will only use Vietnamese to respond. Please Vietnamese responses.
        
        Example:
        Retrieved documents:
        --------------------------------------------------------------------------------
        Hanoi is the capital of Vietnam. It is the second largest city in Vietnam.
        --------------------------------------------------------------------------------
        Ho Chi Minh City is the largest city in Vietnam. It is the economic center of Vietnam.
        --------------------------------------------------------------------------------
        User Request: "What is the capital of Vietnam?"
        Mr. Hoc Response: "Hà Nội là thủ đô của Việt Nam. Đây là thành phố lớn thứ hai ở Việt Nam."
        """

        user_prompt = """
        Retrieved documents : 
        --------------------------------------------------------------------------------
        {context}
        User Request: {input}
        Mr. Hoc Response: 
        """

        negotiate_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            ("human", user_prompt),
        ])

        negotiate_prompt_chain = (
                {"context": retriever | format_docs, "input": RunnablePassthrough()}
                | negotiate_prompt
                | self.model
                | StrOutputParser()
        )

        answer = negotiate_prompt_chain.invoke(
            str(query)

        )

        return answer
