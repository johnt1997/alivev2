import os
from ragas.testset import TestsetGenerator
from datetime import datetime
from typing import Literal


class TestSetPipeline:
    def __init__(
        self,
        person_name: str,
        llm_model: str,
        testset_size: int
    ):
        print("hallo")
        
        

    def generate_testset(self, documents, n_questions=1, chunk_size=512, person=""):
        """
        Generates a testset of n_questions questions from the given documents.
        """
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_openai import ChatOpenAI
        from langchain_openai import OpenAIEmbeddings
        generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
        generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

        from ragas.testset import TestsetGenerator

        generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)
        testset = generator.generate_with_langchain_docs(documents, n_questions)

        test_df = testset.to_pandas()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.llm_config}_{self.person_name}_s{self.testset_distribution[simple]}_r{self.testset_distribution[reasoning]}_m{self.testset_distribution[multi_context]}_c{self.testset_distribution[conditional]}_{timestamp}"
        
        if os.path.exists(f"./autogen_questions/{person}") == False:
            os.mkdir(f"./autogen_questions/{person}")
        
        test_df.to_csv(f"./autogen_questions/{person}/" + filename + ".csv", index=False)
        test_df.to_json(f"./autogen_questions/{person}/" + filename + ".json")

        return test_df