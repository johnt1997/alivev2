# Adapted from https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/chains/chat_vector_db/prompts.py
from langchain_core.prompts.prompt import PromptTemplate


_template = """<s>[INST] <<SYS>>
Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
If the follow up question is already a standalone question, just copy it as is.

Here are some examples on how a question would be reformulated to a standalone question:

Example #1
Human: When were you born?
AI:  I was born on March 14, 1879.
Follow Up Input:  Where?
Standalone question: Where were you born?

Example #2
Human: What is your favorite color?
AI: My favorite color is blue.
Follow Up Input: Why?
Standalone question: Why is your favorite color blue?

Example #3
Human: What did the horse eat from the bowl?
AI: The horse ate the oats from the bowl.
Follow Up Input: How?
Standalone question: How did the horse eat the oats from the bowl?

Here are some examples on questions that would already be standalone questions:

Example #1
Human: What is your name?
AI: My name is Albert Einstein.
Follow Up Input: Where were you born?
Standalone question: Where were you born?

Example #2
Human: Have you ever received a Nobel Price?
AI: Yes, I have received a Nobel Price.
Follow Up Input: Have you ever been married?
Standalone question: Have you ever been married?

Example #3
Human: What is the capital of France?
AI: The capital of France is Paris.
Follow Up Input: What is the population of Paris?
Standalone question: What is the population of Paris?


<</SYS>>
Chat History:
{chat_history}

Follow Up Question: {question}
Standalone question:  [/INST]"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)
