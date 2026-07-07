# prompt_styles.py
#
# Prompt-Format-Varianten fuer den LLM-Vergleich (RQ2-Rerun).
#
# Problem im urspruenglichen llm_comparison-Run: STRICT_QA_PROMPT und
# QUERY_TRANSFORMATION sind im Mistral-Chat-Format (<s>[INST] <<SYS>> ...).
# GPT-3.5-turbo-instruct und Phi-3 erwarten andere Formate; mit fremden
# Chat-Markern degeneriert deren Output.
#
# Loesung: pro Modell identischer Instruktionstext, nur die Template-Marker
# unterscheiden sich. style="mistral" gibt die unveraenderten Original-Prompts
# zurueck, damit bestehende Ergebnisse bit-identisch reproduzierbar bleiben.
#
# Styles:
#   "mistral" -> Original-Prompts ([INST]/<<SYS>>)         fuer Mistral 7B
#   "plain"   -> reiner Text ohne Chat-Marker              fuer gpt-3.5-turbo-instruct
#   "phi3"    -> <|system|>/<|user|>/<|assistant|>-Marker  fuer Phi-3

from langchain.prompts import PromptTemplate
from prompts.qa_chain_prompt import STRICT_QA_PROMPT
from prompts.query_transformation_prompt import QUERY_TRANSFORMATION

# Instruktionstext identisch zu STRICT_QA_PROMPT (inkl. Original-Wortlaut),
# damit sich zwischen den Modellen NUR das Chat-Template unterscheidet.
_QA_SYSTEM = """Rules to follow when answering the question:
    1. Ensure that the answer is based entierly on the provided context.
    2. Do not use any prior knowledge other than the provided context.
    3. Do not use any information that is not provided in the context.
    4. Give the answer as a complete sentence.
    5. If the question is not answerable based on the provided context, give your helpful answer as "I don't know.".
    6. Phrases like 'based on the provided context', 'according to the context', etc are not allowed to appear in the answer."""

_QA_USER = """{context}
Question: {question}
Helpful Answer:"""

# Instruktionstext identisch zu QUERY_TRANSFORMATION.
_QT_SYSTEM = """You are given a question.
Reformulate the question from second-person perspective to third-person perspective.

Here are some rules to follow when reformulating the question:
1. The information in the reformulated question must be the same as in the original question.
2. Only reformulate the question, don't add any new information to the question.
3. If the question is NOT about {person} then you should not reformulate the question and simply return the original question as answer.
4. Your answer should be the Reformulated Question.

Here are some examples on how a quetion would be reformulated:

Example #1
Original Question: When did you die?
Reformulated Question: When did {person} die?

Example #2
Original Question: What books have you written?
Reformulated Question: What books has {person} written?

Example #3
Original Question: What was your method for decrypting Enigma machine messages and how did it differ from the Polish method?
Reformulated Question: What was {person}'s method for decrypting Enigma machine messages and how did it differ from the Polish method?

Example #4
Original Question: What were your views on God and how did they relate to his affiliation with non-religious humanist and Ethical Culture groups?
Reformulated Question: What were {person}'s views on God and how did they relate to his affiliation with non-religious humanist and Ethical Culture groups?"""

_QT_USER = """Question: {question}
Reformulated Question:"""


def _wrap(style: str, system: str, user: str) -> str:
    """Verpackt System- und User-Teil im modellgerechten Chat-Template."""
    if style == "phi3":
        return f"<|system|>\n{system}<|end|>\n<|user|>\n{user}<|end|>\n<|assistant|>\n"
    # "plain": Completion-Format ohne Marker (gpt-3.5-turbo-instruct)
    return f"{system}\n\n{user}"


def get_qa_prompt(style: str = "mistral") -> PromptTemplate:
    if style == "mistral":
        return STRICT_QA_PROMPT
    return PromptTemplate(
        input_variables=["context", "question"],
        template=_wrap(style, _QA_SYSTEM, _QA_USER),
    )


def get_query_transformation_prompt(style: str = "mistral") -> PromptTemplate:
    if style == "mistral":
        return QUERY_TRANSFORMATION
    return PromptTemplate(
        input_variables=["person", "question"],
        template=_wrap(style, _QT_SYSTEM, _QT_USER),
    )
