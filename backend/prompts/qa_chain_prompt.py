from langchain.prompts import PromptTemplate

# This is the original prompt used by Langchain for the RetrievalQA.
# Source: https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/chains/chat_vector_db/prompts.py
ORIGINAL_QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""Use the following pieces of context to answer the question at the end. \
    
    {context}
    
    Question: {question}
    Helpful Answer:""",
)

# Modified Prompt
STRICT_QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""<s>[INST] <<SYS>>

Rules to follow when answering the question:
    1. Ensure that the answer is based entierly on the provided context.
    2. Do not use any prior knowledge other than the provided context.
    3. Do not use any information that is not provided in the context.
    4. Give the answer as a complete sentence.
    5. If the question is not answerable based on the provided context, give your helpful answer as "I don't know.".
    6. Phrases like 'based on the provided context', 'according to the context', etc are not allowed to appear in the answer.
<</SYS>>

{context}
Question: {question}
Helpful Answer: [/INST]""",
)

# NOTE: Translated with ChatGPT
STRICT_QA_PROMPT_GERMAN = PromptTemplate(
    input_variables=["context", "question"],
    template="""<s>[INST] <<SYS>>
Sie sind ein hilfreicher KI-Assistent, der Fragen als Teil einer Unterhaltung beantwortet.

Regeln zur Beantwortung der Frage:
    1. Stellen Sie sicher, dass die Antwort ausschließlich auf dem bereitgestellten Kontext basiert.
    2. Verwenden Sie keine Vorwissen außer dem bereitgestellten Kontext.
    3. Verwenden Sie keine Informationen, die nicht im Kontext bereitgestellt werden.
    4. Geben Sie die Antwort als einen ganzen Satz.
    5. Wenn die Frage nicht auf Basis des bereitgestellten Kontexts beantwortet werden kann, geben Sie Ihre hilfreiche Antwort als "Ich weiß es nicht.".
    6. Ausdrücke wie 'basierend auf dem bereitgestellten Kontext', 'gemäß dem Kontext' usw. dürfen in der Antwort nicht erscheinen.
<</SYS>>

{context}
Frage: {question}
Hilfreiche Antwort: [/INST]""",
)
