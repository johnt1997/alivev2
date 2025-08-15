from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import (
    StreamingStdOutCallbackHandler,
)
from packages.globals import (
    TEMPERATURE,
    MODEL_PATH,
    N_CTX,
    VERBOSE,
    N_GPU_LAYERS,
    N_BATCH,
    MODEL_NAME,
    F16_KV,
)
from langchain.llms.llamacpp import LlamaCpp
from langchain.llms.openai import OpenAI
from dotenv import load_dotenv
import os


class LLMConfig:
    def __init__(
        self,
        temperature=TEMPERATURE,
        model_path=MODEL_PATH,
        model_name=MODEL_NAME,
        n_ctx=N_CTX,
        verbose=VERBOSE,
        n_gpu_layers=N_GPU_LAYERS,
        n_batch=N_BATCH,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        f16_fk=F16_KV,
    ):
        self.temperature = temperature
        self.model_path = model_path
        self.model_name = model_name
        self.n_ctx = n_ctx
        self.verbose = verbose
        self.n_gpu_layers = n_gpu_layers
        self.n_batch = n_batch
        self.callback_manager = callback_manager
        self.f16_kv = f16_fk

    def get_local_llm(self, use_openai: bool = False) -> LlamaCpp:
        if use_openai:
            load_dotenv()
            openai_api_key = os.getenv("OPENAI_API_KEY")
            llm = OpenAI(openai_api_key=openai_api_key, temperature=self.temperature)
            return llm

        print(self.model_path + self.model_name)
        llm = LlamaCpp(
            temperature=self.temperature,
            model_path=self.model_path + self.model_name,
            n_ctx=self.n_ctx,
            n_batch=self.n_batch,
            verbose=self.verbose,
            f16_kv=self.f16_kv,
            n_gpu_layers=self.n_gpu_layers,
            # callback_manager=self.callback_manager,
        )
        return llm
