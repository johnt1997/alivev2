# History-Chat-Bot

The Artificial Legacy in Virtual Environments (ALiVE) history chatbot was developed as part of the master thesis *"Dialogues Across Time: Impersonating Historical Figures with Large Language Models and Retrieval Augmented Generation"*. The general idea and functionality of the ALiVE system is to provide users with an immersive history education learning experience. The ALiVE system allows for document uploads that are the knowledge base for a Large Language Model (LLM) driven Retrieval Augmented Generation (RAG) pipeline. With that, users can have conversations with historical figures upon creating them within the ALiVE system.

## Before Installation

- Make sure you have [Git LFS](https://git-lfs.com/) installed before cloning.
- Download [Docker](https://www.docker.com/products/docker-desktop/) and start the desktop application.
- Download the [Models](https://drive.google.com/file/d/1BjTGTWeEu6AGplxkfl3dM1y3pfoi7YC2/view?usp=sharing).

By downloading and using the models you agree to the terms and conditions for:

- [mistral.ai](https://mistral.ai/)
  - [Quantified Model](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/blob/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf) LLM Model
  - [Original Model](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)
- [coqui.ai](https://coqui.ai/cpml) TTS Model
- [Beijing Academy of Artificial Intelligence](https://huggingface.co/BAAI/bge-small-en-v1.5) Embedding Model

### Setup Model Folder Structure

After downloading the models, extract and add the files to the **backend/models** folder. The folder structure should look like this when done so:

NOTE: The files and folder MUST be named as seen below.

```bash
└── backend
    ├── ....
    ├── models
    │   ├── bge-small-en-v1.5
    │   │   └── ...
    │   ├── tts_models--multilingual--multi-dataset--xtts_v2
    │   │   └── ...
    │   └── mistral-7b-instruct-v0.1.Q4_K_M.gguf
    │
    ├── ....
    ├── .env
    └── ....
```

### OpenAI

If you want to use OpanAI please add you API key to the .env file inside the **backend/** folder:

```bash
OPENAI_API_KEY=<your-openai-api-key>
```

NOTE: Add your key BEFORE building the docker container!

# RUN THE APPLICATION WITH DOCKER

Start Docker and then inside the **root** project folder run:

```bash
docker-compose up
# The webapplication can then be accessed under localhost:3000
# The server can then be accessed under localhost:8080
# after finishing you can run docker compose down to stop the containers
```

### Altertatively: Inside the **/backend** folder run:

```bash
# Backend Container
docker volume create <my-vol-1>
docker volume create <my-vol-2>
docker build --tag <docker-name-1> .
docker run --rm -p 8080:8080 -v <my-vol-1>:/root/.local/share/tts -v <my-vol-2>:/app -it <docker-name-1>
```

Inside the **/frontend/history-chat-bot** folder

```bash
# Frontend Container
docker build --tag <docker-name-2> .
docker run --rm -p 3000:3000 -it <docker-name-2>
```

# Performance Notice

> **Warning: Response times depend heavily on your hardware.**
>
> - **First request after startup:** The system builds a vector database from your documents and loads the LLM into memory. This can take **10–25 minutes** depending on document size, chosen chunking strategy, and hardware.
> - **Subsequent requests:** Vector database is cached; only LLM inference runs. Still expect **5–15 minutes per response** on CPU-only machines.
> - **Semantic chunking** (default) is the slowest strategy but produces the best retrieval quality. Switch to *Recursive* in the sidebar for faster initial loading.
> - **GPU acceleration** is only supported for NVIDIA (CUDA) and Apple Silicon (Metal). Intel integrated graphics do **not** accelerate inference.
> - **Recommended for faster responses:** Enable the *Use OpenAI* toggle in the sidebar and provide an `OPENAI_API_KEY` in `backend/.env`. This offloads LLM inference to the cloud and reduces response time to a few seconds.

# Usage

Read the remarks on usage before using the tool.

### Language

- The prefered language to use is English. This regards the documents uploaded as well as the language on which questions should be asked.
- However, one can upload documents in German and ask the questions in German. In this case the answer is also provided in German. However to do so, set the language to German in the UI and press the **Apply** button (left side of landing page). Consider this functionality as experimental.
- Though possible, it is best to not mix Document Language, Question Language and Output Language eg. If the documents are in english, ask in english and have the language set to english in the sidebar.

### Output

- Questions are answered based on the provided context, retrieved from the uploaded documents for a person.
- As the context is retrieved via simmilarity search, the more specific a question is, the better the retrieved context and the answers are. If the question cant be answered based on the retrieved context, the person will most likely answer with "I don't know."
- The context can be seen in the accordeon element above the textinput. It features the source, the context and the simmilarity score, ranging from 0-1 with 1 being the most simmilar. A low score indicates that the context retrieved does not match the question and thus is most likely not suitable to answer the question.
- Even though the aim of Retrieval Augmented Generation, as used in this implementation, is to provide answers based solely on provided context, halluicination might still occure.

### Creating a Person

- Name is mandatory
  - NOTE: Creating a person with the same name as a existing one will overwrite the old one and its data.
- Only .pdf and .txt files are allowed for datafiles
- The personality (optional) defines how a person would give a answer. Add 2-3 examples. This should be enough. Here is an example for William Shakespeare:

  - **Original Answer:**
    It is not exactly known when I was born but it is assumed that it was on April 23, 1564.
  - **Reformulated Answer:**
    Upon the stage of life, I made my entrance on the twenty-third day of April in the year 1564, beneath the heavens that grace Stratford-upon-Avon with their eternal vigil.

- Voice Upload (optional): use .wav files for voice upload
- Virtual Characters (optional) can be created using [ReadyPlayerMe](https://readyplayer.me/)
- Persons can also be deleted: Note that your data will be gone if once deleted
- Persons are editable

# Optional Installation

## Prerequieries

- [Conda](https://docs.conda.io/projects/miniconda/en/latest/)
- [Python](https://www.python.org/)
- [Node.js v21.2.0](https://nodejs.org/en)
- [espeak](https://espeak.sourceforge.net/)

After installation of the prerequieries:

```bash
conda create -n <env_name> python=3.9
conda activate <env_name>
```

Inside the **base** folder of the project run the following commands:

```bash
pip3 install -r requirements.txt
```

Additionally check [llama-cpp-python](https://python.langchain.com/docs/integrations/llms/llamacpp) for addidional installation information.
On Mac and Linux:

```bash
CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 pip install llama-cpp-python==0.2.38
```

### Start the Server

Inside the **backend/** folder

```bash
python3 server.py
# server running on localhost:8080
```

### Start the UI

Inside the **frontend/history-chat-bot/** folder

```bash
npm i
npm run build
npm run start
# frontend running on localhost:3000
```

# RAG Pipeline Evaluation

The repository includes a complete evaluation framework for comparing different RAG pipeline configurations. All experimental results reported in the master's thesis can be reproduced from the artefacts in this repository.

### 1. Generating Evaluation Questions

For a new corpus (PDFs in `backend/data/<corpus>/`):

```bash
cd backend
python create_questions.py --corpus Eu --output hilfreich2.csv --n 42
```

The Washington question set was generated previously with the RAGAS `TestsetGenerator` (single-hop, multi-hop specific, and multi-hop abstract synthesizers) and is checked in at `backend/autogen_questions/Washington/hilfreich.csv`. The EU question set in `backend/autogen_questions/Eu/hilfreich2.csv` was generated with `create_questions.py` (GPT-4o-mini).

### 2. Running Evaluations

```bash
cd backend

# Run all 12 pipeline combinations (3 chunkers × 4 retrieval modes)
python run_evaluation.py --corpus Washington

# Run specific pipelines only
python run_evaluation.py --corpus Washington --pipelines dense_semantic hybrid_semantic_rerank

# LLM comparison (e.g., GPT-3.5 instead of Mistral)
python run_evaluation.py --corpus Washington --pipelines dense_semantic --llm gpt35 --output-dir llm_comparison

# Chunk-size experiment
python run_evaluation.py --corpus Washington --pipelines dense_recursive --chunk-sizes 500 1000 1500 --output-dir chunk_size_experiment

# Cross-corpus replication on EU data
python run_evaluation.py --corpus Eu

# Quick test with only 3 questions
python run_evaluation.py --corpus Washington --limit 3
```

### Pipeline Configurations

| Chunker | Retrieval Modes |
|---------|-----------------|
| `recursive` | dense, dense+rerank, hybrid, hybrid+rerank |
| `sentence_transformer` | dense, dense+rerank, hybrid, hybrid+rerank |
| `semantic` | dense, dense+rerank, hybrid, hybrid+rerank |

### Evaluation Metrics (RAGAS)

- **Faithfulness**: Is the answer grounded in the retrieved context?
- **Answer Relevancy**: Does the answer address the question?
- **Context Precision**: Are the retrieved chunks relevant?
- **Context Recall**: Are all necessary facts retrieved?
- **Answer Correctness**: Does the answer match the ground truth?
- **Semantic Similarity**: Embedding similarity to ground truth

All RAGAS-based metrics use OpenAI's `gpt-3.5-turbo-16k` as the judge model (RAGAS v0.1.3 default).

### 3. Analysis & Reproducibility

After running evaluations, use `mega_analysis_notebook.ipynb` to generate:
- Aggregate statistics per pipeline (means, std)
- Statistical significance tests (paired Wilcoxon signed-rank with Holm correction, rank-biserial effect sizes)
- Pearson-correlation analysis of RAGAS metrics
- Visualizations: heatmaps, per-question-type bars, cross-corpus comparison charts

For the cross-corpus comparison plot reported in Chapter 5 of the thesis, run:

```bash
cd backend
python cross_corpus_plot.py
```

### Result Directory Layout

```
backend/results/
├── Washington/
│   ├── final_run_42Q/          # Main 12-pipeline evaluation (Chapter 5 results)
│   ├── chunk_size_experiment/  # Robustness check (500/1000/1500 chunks)
│   ├── llm_comparison/         # Mistral vs. GPT-3.5 vs. Phi-3
│   └── final_run_slm/          # Small-LM comparison
├── Eu/
│   └── final_run/              # Cross-corpus replication
├── results_all/                # Aggregated copy of all final runs
└── cross_corpus_comparison.png # Plot used in the thesis
```

Each pipeline run produces two CSVs: `<pipeline>_answers_raw.csv` (per-question outputs) and `<pipeline>_final_results.csv` (per-question RAGAS scores plus configuration metadata).

# Complete Project Structure

```bash
History-Chat-Bot
├── backend
│   ├── data                    # PDF documents per person
│   ├── database                # ChromaDB vector stores (auto-generated)
│   ├── models                  # Download and set Models as described above
│   │   ├── bge-small-en-v1.5
│   │   ├── mistral-7b-instruct-v0.1.Q4_K_M.gguf
│   │   └── tts_models--multilingual--multi-dataset--xtts_v2
│   ├── packages                # Core RAG pipeline code
│   ├── persons                 # Person configurations
│   ├── prompts                 # LLM prompt templates
│   ├── autogen_questions       # Evaluation question sets
│   ├── results                 # Evaluation results (CSV)
│   ├── bm25_jsonl              # BM25 indices for hybrid search
│   ├── .env
│   ├── server.py               # Backend API server
│   ├── run_evaluation.py       # Evaluation script
│   ├── mega_analysis_notebook.ipynb  # Analysis notebook
│   └── requirements.txt
│
├── frontend
│   └── history-chat-bot
│
├── compose.yaml
├── README.md
└── requirements.txt
```
