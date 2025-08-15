from langchain.prompts import PromptTemplate

QUERY_TRANSFORMATION = PromptTemplate(
    input_variables=["person", "question"],
    template="""<s>[INST] <<SYS>>
You are given a question.
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
Reformulated Question: What were {person}'s views on God and how did they relate to his affiliation with non-religious humanist and Ethical Culture groups?


<</SYS>>


Question: {question}
Reformulated Question: [/INST]""",
)

# NOTE: Translated with ChatGPT
QUERY_TRANSFORMATION_GERMAN = PromptTemplate(
    input_variables=["person", "question"],
    template="""<s>[INST] <<SYS>>
Sie erhalten eine Frage über {person}.
Formulieren Sie die Frage aus der Perspektive der dritten Person um.

Hier sind einige Regeln zu beachten, wenn Sie die Frage umformulieren:
1. Die Informationen in der umformulierten Frage müssen dieselben sein wie in der Originalfrage.
2. Formulieren Sie nur die Frage um, fügen Sie keine neuen Informationen hinzu.
3. Ihre Antwort sollte die umformulierte Frage sein.

Hier sind einige Beispiele, wie eine Frage umformuliert werden würde:

Beispiel #1
Originalfrage: Wann sind Sie gestorben?
Umformulierte Frage: Wann ist {person} gestorben?

Beispiel #2
Originalfrage: Welche Bücher haben Sie geschrieben?
Umformulierte Frage: Welche Bücher hat {person} geschrieben?

Beispiel #3
Originalfrage: Was war Ihre Methode zur Entschlüsselung von Enigma-Nachrichten und wie unterschied sie sich von der polnischen Methode?
Umformulierte Frage: Was war die Methode von {person} zur Entschlüsselung von Enigma-Nachrichten und wie unterschied sie sich von der polnischen Methode?

Beispiel #4
Originalfrage: Was waren Ihre Ansichten über Gott und wie standen sie in Beziehung zu seiner Zugehörigkeit zu nicht-religiösen humanistischen und ethischen Kulturgemeinschaften?
Umformulierte Frage: Was waren {person}'s Ansichten über Gott und wie standen sie in Beziehung zu seiner Zugehörigkeit zu nicht-religiösen humanistischen und ethischen Kulturgemeinschaften?


<</SYS>>


Frage: {question}
Umformulierte Frage: [/INST]""",
)
