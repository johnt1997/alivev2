from langchain.llms.llamacpp import LlamaCpp
from langchain.schema.vectorstore import VectorStoreRetriever
from typing import List, Tuple
from langchain.schema.document import Document
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores.faiss import FAISS
from packages.globals import CHAIN_TYPE, SEARCH_KWARGS_NUM
from prompts.qa_chain_prompt import STRICT_QA_PROMPT, STRICT_QA_PROMPT_GERMAN
from prompts.translation_prompt import TRANSLATE, TRANSLATE_OPENAI
from prompts.impersonation_prompt import IMPERSONATION_PROMPT
from prompts.impersonation_prompt_with_personality import (
    IMPERSONATION_PROMPT_WITH_PERSONALITY,
)
from prompts.query_transformation_prompt import (
    QUERY_TRANSFORMATION,
    QUERY_TRANSFORMATION_GERMAN,
)
from packages.person import Person
from prompts.chat_history import CONDENSE_QUESTION_PROMPT


class InitializeQuesionAnsweringChain:
    def __init__(
        self,
        llm: LlamaCpp,
        retriever: VectorStoreRetriever,
        db: FAISS,
        person: Person,
        chain_type: str = CHAIN_TYPE,
        search_kwargs_num: int = SEARCH_KWARGS_NUM,
        language: str = "en",
    ):
        self.llm = llm
        self.chain_type = chain_type
        self.retriever = retriever
        self.db = db
        self.search_kwargs_num = search_kwargs_num
        self.language = language
        self.person = person
        self.memory = self._init_memory()

    def _init_memory(self):
        """Initializes the memory for the conversational chain. Allows to ask follow up questions during conversation."""
        memory = ConversationBufferMemory(memory_key="chat_history")
        return memory

    def _transform_question(self, query: str, person: Person) -> str:
        prompt = QUERY_TRANSFORMATION
        if self.language == "de":
            prompt = QUERY_TRANSFORMATION_GERMAN
        answer = self.llm.invoke(prompt.format(person=person.name, question=query))
        return answer

    def _get_relevant_docs(self, query: str) -> List[Document]:
        """Returns the relevant documents regarding the user input."""
        return self.db.similarity_search(query, k=self.search_kwargs_num)

    def _get_relevant_docs_with_scores(
        self, query: str
    ) -> List[Tuple[Document, float]]:
        """Returns the relevant documents with scores regarding the user input."""

        print("[INFO] Loading documents with scores...")
        return self.db.similarity_search_with_relevance_scores(
            query, k=self.search_kwargs_num
        )

    def get_answer_with_docs_and_scores(
        self,
        query: str,
        impersonate: bool = True,
        previous_question: str = "",
        previous_answer: str = "",
    ) -> Tuple[Tuple[str, dict], List[Tuple[Document, float]]]:
        """Transform the question, retrieve the documents and get the answer from the conversational chain."""

        original_query = query
        follow_up_question = ""
        third_person_query = ""
        original_answer = ""
        transformed_answer = ""

        dict_query = {
            "original_query": original_query,
            "follow_up_question": follow_up_question,
            "third_person_query": third_person_query,
            "original_answer": original_answer,
            "transformed_answer": transformed_answer,
        }

        print("[INFO] Processing Question...")
        if impersonate:
            if previous_answer != "" and previous_question != "":
                print("[INFO] Adding previous conversation to memory...")
                print("[INFO] Previous Question:", previous_question)
                print("[INFO] Previous Answer", previous_answer)
                self.memory.chat_memory.add_user_message(previous_question)
                self.memory.chat_memory.add_ai_message(previous_answer)
                query = self.llm.invoke(
                    CONDENSE_QUESTION_PROMPT.format(
                        chat_history=self.memory.chat_memory,
                        question=query,
                    )
                )
                follow_up_question = query

            query = self._transform_question(query, person=self.person)
            third_person_query = query
        print("[INFO] Querying Documents with question: " + query)

        relevant_documents = self._get_relevant_docs_with_scores(query)

        strictPrompt = STRICT_QA_PROMPT
        impersonationPrompt = IMPERSONATION_PROMPT
        personalityPrompt = IMPERSONATION_PROMPT_WITH_PERSONALITY
        translationPrompt = TRANSLATE
        if self.language == "de":
            pass
            # strictPrompt = STRICT_QA_PROMPT_GERMAN
            # impersonationPrompt = IMPERSONATION_PROMPT_GERMAN
            # personalityPrompt = IMPERSONATION_PROMPT_WITH_PERSONALITY_GERMAN
        if self.llm.__class__.__name__ == "OpenAI":
            translationPrompt = TRANSLATE_OPENAI

        result = self.llm.invoke(
            strictPrompt.format(question=query, context=relevant_documents)
        )
        print("[INFO] Original Answer: " + result)
        original_answer = result

        if impersonate and self.person.personality is None:
            result = self.llm.invoke(
                impersonationPrompt.format(person=self.person.name, answer=result)
            )
            print("[INFO] Impersonation: " + result)
        if impersonate and self.person.personality is not None:
            result = self.llm.invoke(
                personalityPrompt.format(
                    person=self.person.name,
                    answer=result,
                    personality=self.person.format_personality_prompt(),
                )
            )
            print("[INFO] Impersonation: " + result)

        if self.language == "de":
            result = self.llm.invoke(
                translationPrompt.format(text=result, language="German")
            )
            print("[INFO] Translated: " + result)

        transformed_answer = result
        dict_query = {
            "original_query": original_query,
            "follow_up_question": follow_up_question,
            "third_person_query": third_person_query,
            "original_answer": original_answer,
            "transformed_answer": transformed_answer,
        }

        return (result, dict_query), relevant_documents
