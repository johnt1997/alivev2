from langchain_community.embeddings import HuggingFaceEmbeddings
CHUNK_SIZE=-1
CHUNK_OVERLAP=-1
SEARCH_KWARGS_NUM = 3
SEARCH_TYPE = "similarity"
VECTOR_STORE_PATH = "./database/"
EMBEDDING_MODEL_NAME = "./models/bge-small-en-v1.5"
EMBEDDINGS = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
MODEL_PATH = "./models/"
#MODEL_NAME = "Phi-3-mini-4k-instruct-q4.gguf"
MODEL_NAME = "mistral-7b-instruct-v0.1.Q4_K_M.gguf"
TEMPERATURE = 0.1
N_CTX = 3900
F16_KV = True
VERBOSE = False
N_BATCH = 512
N_GPU_LAYERS = -1
CHAIN_TYPE = "stuff"
