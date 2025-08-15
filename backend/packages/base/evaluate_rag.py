import os
from packages.init_chain import InitializeQuesionAnsweringChain
from prompts.qa_chain_prompt import STRICT_QA_PROMPT
from tqdm import tqdm


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class EvaluationPipeline:
    def __init__(
        self,
        qa_chain: InitializeQuesionAnsweringChain,
        eval_questions: list[str],
        eval_answers: list[str],
        eval_file_name: str,
    ):
        self.llm = None
        self.embedding = None
        self.eval_questions = eval_questions
        self.eval_answers = eval_answers
        self.eval_file_name = eval_file_name
        self.examples = [
            {"question": q, "ground_truths": [self.eval_answers[i]]}
            for i, q in enumerate(self.eval_questions)
        ]

        self.qa_chain: InitializeQuesionAnsweringChain = qa_chain

    def generate_answers(self):
        results_list = []
        for example in tqdm(self.examples):

            relevant_docs = self.qa_chain._get_relevant_docs(query=example["question"])
            result = self.qa_chain.llm.invoke(
                STRICT_QA_PROMPT.format(
                    question=example["question"], context=relevant_docs
                )
            )
            results_list.append(
                {
                    "question": example["question"],
                    "answer": result,
                    "contexts": relevant_docs,
                }
            )

        return results_list
