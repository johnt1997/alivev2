from langchain.prompts import PromptTemplate

# Prompt to translate text into a language of choice.
TRANSLATE = """Translate the following text into {language}. 
Your answer should be the translated text. 

text: {text}
language: {language}

Answer:"""

# Prompt to translate text into a language of choice.
TRANSLATE_OPENAI = """Translate the following text into {language}. 

Here are some rules to follow when translating:
1. If the text is already in the target language, just give the text in its original form as answer. 

text: {text}
language: {language}

Answer:"""
