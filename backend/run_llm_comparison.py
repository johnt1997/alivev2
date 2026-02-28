"""
LLM Comparison Script — runs dense_semantic with GPT-3.5 and Phi-3 as generators.
Keeps retrieval identical (BGE embeddings, FAISS, same chunks) to isolate generator effect.

Usage:
    # GPT-3.5 (local, needs OPENAI_API_KEY in .env):
    python run_llm_comparison.py --llm openai

    # Phi-3 (needs GGUF model in ./models/):
    python run_llm_comparison.py --llm phi3

    # Both sequentially:
    python run_llm_comparison.py --llm both
"""
import os
import sys
import time
import argparse
import gc

os.environ["TOKENIZERS_PARALLELISM"] = "false"
if sys.platform == "win32":
    os.environ["JAVA_HOME"] = r"C:\Program Files\Eclipse Adoptium\jdk-17.0.18.8-hotspot"

import pandas as pd
try:
    import torch
except (ImportError, OSError):
    torch = None
from dotenv import load_dotenv
load_dotenv()

from ragas import evaluate
from datasets import Dataset
from packages.person import Person
from packages.globals import EMBEDDINGS
from packages.evaluate_rag import EvaluationPipeline
from packages.vector_store_handler import VectorStoreHandler
from packages.document_processing import DocumentProcessing
from packages.init_chain import InitializeQuesionAnsweringChain

from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    answer_correctness,
    answer_similarity,
)

# === CONFIG ===
PIPELINE_CONFIG = {
    "splitter_type": "semantic",
    "chunk_size": 0,
    "chunk_overlap": 0,
    "retrieval_mode": "dense",
    "use_reranker": False,
}

name = "Washington"
RESULTS_DIR = f"./results/{name}/llm_comparison"
os.makedirs(RESULTS_DIR, exist_ok=True)

metric_columns = [
    'context_precision', 'faithfulness', 'answer_relevancy',
    'context_recall', 'answer_correctness', 'answer_similarity'
]


def get_llm(llm_type):
    """Load the specified LLM."""
    if llm_type == "openai":
        from langchain.llms.openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("[ERROR] OPENAI_API_KEY not found in .env")
            sys.exit(1)
        llm = OpenAI(openai_api_key=api_key, temperature=0.0, model_name="gpt-3.5-turbo-instruct")
        print("[OK] Loaded GPT-3.5-turbo-instruct")
        return llm, "gpt35"

    elif llm_type == "phi3":
        from langchain.llms.llamacpp import LlamaCpp
        import packages.globals as g
        model_path = g.MODEL_PATH + "Phi-3-mini-4k-instruct-q4.gguf"
        if not os.path.exists(model_path):
            print(f"[ERROR] Phi-3 model not found: {model_path}")
            sys.exit(1)
        llm = LlamaCpp(
            temperature=0.0,
            model_path=model_path,
            n_ctx=3900,
            n_batch=512,
            verbose=False,
            f16_kv=True,
            n_gpu_layers=g.N_GPU_LAYERS,
        )
        print(f"[OK] Loaded Phi-3 (GPU layers: {g.N_GPU_LAYERS})")
        return llm, "phi3"

    else:
        raise ValueError(f"Unknown LLM type: {llm_type}")


def run_pipeline(llm, llm_label, eval_questions, ground_truths, question_types):
    """Run dense_semantic pipeline with the given LLM."""
    output_path = f"{RESULTS_DIR}/dense_semantic_{llm_label}_final_results.csv"

    if os.path.exists(output_path):
        print(f"[SKIP] {output_path} already exists")
        return

    print(f"\n{'='*60}")
    print(f"Running dense_semantic with {llm_label}")
    print(f"{'='*60}")

    pipeline_start = time.time()

    # Setup retriever (identical for all LLMs)
    person = Person(name=name)
    embedding = EMBEDDINGS
    search_kwargs_num = 3

    vectorstore_handler = VectorStoreHandler(
        embeddings=embedding, search_kwargs_num=search_kwargs_num
    )

    database_path = vectorstore_handler._get_vector_store_path(
        vector_store_name=person.name,
        splitter_type=PIPELINE_CONFIG["splitter_type"],
        chunk_size=PIPELINE_CONFIG["chunk_size"],
        chunk_overlap=PIPELINE_CONFIG["chunk_overlap"]
    )
    print(f"  FAISS path: {database_path}")

    if not os.path.exists(database_path):
        print(f"  Building FAISS DB...")
        doc_processor = DocumentProcessing(embeddings=embedding)
        split_documents = doc_processor.get_chunked_documents(
            directory_path=f"./data/{person.name}",
            splitter_type=PIPELINE_CONFIG["splitter_type"],
            chunk_size=PIPELINE_CONFIG["chunk_size"],
            chunk_overlap=PIPELINE_CONFIG["chunk_overlap"],
        )
        _, db = vectorstore_handler.create_db_and_retriever(
            chunked_documents=split_documents,
            vector_store_name=person.name,
            splitter_type=PIPELINE_CONFIG["splitter_type"],
            chunk_size=PIPELINE_CONFIG["chunk_size"],
            chunk_overlap=PIPELINE_CONFIG["chunk_overlap"],
        )
    else:
        print(f"  Loading existing FAISS DB...")
        _, db = vectorstore_handler.get_db_and_retriever(
            vector_store_name=person.name,
            splitter_type=PIPELINE_CONFIG["splitter_type"],
            chunk_size=PIPELINE_CONFIG["chunk_size"],
            chunk_overlap=PIPELINE_CONFIG["chunk_overlap"],
        )

    if db is None:
        print(f"  [ERROR] Database loading failed.")
        return

    retriever = db.as_retriever(search_kwargs={"k": search_kwargs_num})

    # QA Chain with the specified LLM
    qa_chain = InitializeQuesionAnsweringChain(
        llm=llm, retriever=retriever, db=db, person=person,
        search_kwargs_num=search_kwargs_num, use_reranker=False
    )

    # Generate answers
    print(f"[1/3] Generating answers with {llm_label}...")
    eval_pipeline = EvaluationPipeline(qa_chain=qa_chain, eval_questions=eval_questions)
    generated_answers = eval_pipeline.generate_answers_with_metadata()

    if not generated_answers:
        print(f"  [ERROR] No answers generated.")
        return

    df_results = pd.DataFrame(generated_answers)
    df_results["question_type"] = question_types
    df_results["generator_llm"] = llm_label
    gen_time = time.time() - pipeline_start
    print(f"[OK] {len(df_results)} answers in {gen_time:.0f}s ({gen_time/len(eval_questions):.1f}s/question)")

    # Save raw backup
    backup_path = f"{RESULTS_DIR}/dense_semantic_{llm_label}_answers_raw.csv"
    df_results.to_csv(backup_path, index=False)
    print(f"[BACKUP] Raw answers saved")

    # RAGAS scoring
    print(f"[2/3] RAGAS scoring...")
    df_results['ground_truth'] = ground_truths
    df_results['ground_truths'] = df_results['ground_truth'].apply(lambda x: [x])
    ragas_dataset = Dataset.from_pandas(df_results)
    ragas_results = evaluate(
        ragas_dataset,
        metrics=[context_precision, faithfulness, answer_relevancy,
                 context_recall, answer_correctness, answer_similarity]
    )
    df_ragas_scores = ragas_results.to_pandas()
    df_scores_only = df_ragas_scores[metric_columns]
    final_df = pd.concat([df_results.reset_index(drop=True), df_scores_only.reset_index(drop=True)], axis=1)
    if 'ground_truth' in final_df.columns:
        final_df = final_df.drop(columns=['ground_truth'])

    # Save
    print(f"[3/3] Saving...")
    final_df.to_csv(output_path, index=False)
    print(f"  -> {output_path}")

    total_time = time.time() - pipeline_start
    f = final_df['faithfulness'].mean()
    ar = final_df['answer_relevancy'].mean()
    print(f"  F={f:.3f} AR={ar:.3f} | {total_time:.0f}s total")

    # Cleanup
    gc.collect()
    if torch and torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Comparison on dense_semantic pipeline")
    parser.add_argument("--llm", choices=["openai", "phi3", "both"], default="openai",
                        help="Which LLM to use as generator")
    args = parser.parse_args()

    # Load evaluation questions
    df_eval = pd.read_csv(f'./autogen_questions/{name}/hilfreich.csv')
    eval_questions = df_eval["question"].tolist()
    question_types = df_eval["question_type"].tolist()
    ground_truths = df_eval['ground_truth'].tolist()
    print(f"[OK] Loaded {len(eval_questions)} questions")

    llms_to_run = []
    if args.llm in ("openai", "both"):
        llms_to_run.append("openai")
    if args.llm in ("phi3", "both"):
        llms_to_run.append("phi3")

    for llm_type in llms_to_run:
        llm, label = get_llm(llm_type)
        run_pipeline(llm, label, eval_questions, ground_truths, question_types)

    print(f"\n{'='*60}")
    print("LLM COMPARISON DONE")
    print(f"Results in: {RESULTS_DIR}/")
    print(f"{'='*60}")
