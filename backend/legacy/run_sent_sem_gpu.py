"""
GPU Evaluation Script — runs 8 sentence + semantic pipelines + RAGAS scoring.
Crash-safe: skips pipelines whose final CSV already exists.

Usage on GPU VM:
    cd backend
    python run_sent_sem_gpu.py
"""
import os
import sys
import time
import gc

# === CONFIG ===
GPU_LAYERS = -1
BUILD_BM25 = True

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
if sys.platform == "win32":
    os.environ["JAVA_HOME"] = r"C:\Program Files\Eclipse Adoptium\jdk-17.0.18.8-hotspot"

import packages.globals as g
g.N_GPU_LAYERS = GPU_LAYERS
print(f"[CONFIG] N_GPU_LAYERS = {GPU_LAYERS}")

import torch
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from ragas import evaluate
from datasets import Dataset
from packages.person import Person
from packages.globals import EMBEDDINGS
from packages.llm_config import LLMConfig
from packages.evaluate_rag import EvaluationPipeline
from packages.vector_store_handler import VectorStoreHandler
from packages.document_processing import DocumentProcessing
from packages.bm25_retriever import BM25Retriever
from packages.vector_store_handler import HybridRetriever
from packages.init_chain import InitializeQuesionAnsweringChain

from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    answer_correctness,
    answer_similarity,
)

# === STEP 0: Build BM25 indices for sentence_transformer and semantic ===
name = "Washington"

if BUILD_BM25:
    import subprocess
    for splitter in ["sentence_transformer", "semantic"]:
        index_dir = f"./bm25_indexes/{name}_{splitter}"
        if os.path.isdir(index_dir):
            print(f"[SKIP] BM25 index exists: {index_dir}")
            continue
        print(f"\n[STEP 0] Building BM25 for {splitter}...")
        # Step 1: Build JSONL
        jsonl_dir = f"./bm25_jsonl/{name}_{splitter}"
        if splitter == "sentence_transformer":
            cs, co = "256", "30"
        else:
            cs, co = "0", "0"
        cmd_jsonl = [
            sys.executable, "make_bm25_jsonl.py",
            "--person", name,
            "--splitter", splitter,
            "--chunk_size", cs,
            "--chunk_overlap", co,
        ]

        print(f"  [1/2] Building JSONL...")
        rc = subprocess.run(cmd_jsonl).returncode
        if rc != 0:
            print(f"  [WARN] JSONL build failed for {splitter}")
            continue

        # Step 2: Build Lucene index
        print(f"  [2/2] Building Lucene index...")
        try:
            from pyserini.index.lucene import LuceneIndexer
            indexer = LuceneIndexer(index_dir)
            import json
            docs_path = os.path.join(jsonl_dir, "docs.jsonl")
            with open(docs_path, "r", encoding="utf-8") as f:
                for line in f:
                    doc = json.loads(line)
                    indexer.add_doc_raw(doc)
            indexer.close()
            print(f"  [OK] Index ready: {index_dir}")
        except Exception as e:
            print(f"  [FAIL] Index build failed for {splitter}: {e}")
            # Fallback: use pyserini CLI
            cmd_index = [
                sys.executable, "-m", "pyserini.index.lucene",
                "--collection", "JsonCollection",
                "--input", jsonl_dir,
                "--index", index_dir,
                "--generator", "DefaultLuceneDocumentGenerator",
                "--threads", "1",
                "--storePositions", "--storeDocvectors", "--storeRaw",
            ]
            rc2 = subprocess.run(cmd_index).returncode
            if rc2 == 0:
                print(f"  [OK] Index ready (CLI fallback): {index_dir}")
            else:
                print(f"  [FAIL] Index build failed completely for {splitter}")

# === SETUP ===
RESULTS_DIR = f"./results/{name}/final_run_42Q"
os.makedirs(RESULTS_DIR, exist_ok=True)

print("\n[SETUP] Loading Mistral 7B...")
llm_config = LLMConfig(temperature=0.0)
llm = llm_config.get_local_llm(use_openai=False)
print("[OK] LLM loaded")

df_eval = pd.read_csv(f'./autogen_questions/{name}/hilfreich.csv')
eval_questions = df_eval["question"].tolist()
question_types = df_eval["question_type"].tolist()
ground_truths = df_eval['ground_truth'].tolist()
print(f"[OK] Loaded {len(eval_questions)} questions")


def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_single_test(name, config, questions, llm):
    person = Person(name=name)
    embedding = EMBEDDINGS
    search_kwargs_num = 3

    splitter_type = config["splitter_type"]
    chunk_size = config["chunk_size"]
    chunk_overlap = config["chunk_overlap"]
    retrieval_mode = config["retrieval_mode"]
    use_reranker = config["use_reranker"]

    vectorstore_handler = VectorStoreHandler(
        embeddings=embedding, search_kwargs_num=search_kwargs_num
    )

    database_path = vectorstore_handler._get_vector_store_path(
        vector_store_name=person.name,
        splitter_type=splitter_type,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    print(f"  FAISS path: {database_path}")

    if not os.path.exists(database_path):
        print(f"  Building FAISS DB...")
        doc_processor = DocumentProcessing(embeddings=embedding)
        split_documents = doc_processor.get_chunked_documents(
            directory_path=f"./data/{person.name}",
            splitter_type=splitter_type,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        _, db = vectorstore_handler.create_db_and_retriever(
            chunked_documents=split_documents,
            vector_store_name=person.name,
            splitter_type=splitter_type,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    else:
        print(f"  Loading existing FAISS DB...")
        _, db = vectorstore_handler.get_db_and_retriever(
            vector_store_name=person.name,
            splitter_type=splitter_type,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    if db is None:
        print(f"  [ERROR] Database creation/loading failed.")
        return []

    if retrieval_mode == 'hybrid':
        bm25_index_dir = f"./bm25_indexes/{name}_{splitter_type}"
        print(f"  HYBRID | BM25: {bm25_index_dir} | exists: {os.path.isdir(bm25_index_dir)}")
        if not os.path.isdir(bm25_index_dir):
            print(f"  [ERROR] BM25 index not found!")
            return []
        bm25_retriever = BM25Retriever(bm25_index_dir)
        retriever = HybridRetriever(db=db, bm25_retriever=bm25_retriever, k=search_kwargs_num)
    else:
        print("  DENSE Retriever")
        retriever = db.as_retriever(search_kwargs={"k": search_kwargs_num})

    qa_chain = InitializeQuesionAnsweringChain(
        llm=llm, retriever=retriever, db=db, person=person,
        search_kwargs_num=search_kwargs_num, use_reranker=use_reranker
    )

    eval_pipeline = EvaluationPipeline(qa_chain=qa_chain, eval_questions=questions)
    return eval_pipeline.generate_answers_with_metadata()


# === PIPELINE MATRIX ===
evaluation_matrix = [
    # Sentence Transformer (256 tokens, 30 overlap)
    {"pipeline_name": "dense_sentence",         "splitter_type": "sentence_transformer", "chunk_size": 256, "chunk_overlap": 30, "retrieval_mode": "dense",  "use_reranker": False},
    {"pipeline_name": "dense_sentence_rerank",  "splitter_type": "sentence_transformer", "chunk_size": 256, "chunk_overlap": 30, "retrieval_mode": "dense",  "use_reranker": True},
    {"pipeline_name": "hybrid_sentence",        "splitter_type": "sentence_transformer", "chunk_size": 256, "chunk_overlap": 30, "retrieval_mode": "hybrid", "use_reranker": False},
    {"pipeline_name": "hybrid_sentence_rerank", "splitter_type": "sentence_transformer", "chunk_size": 256, "chunk_overlap": 30, "retrieval_mode": "hybrid", "use_reranker": True},
    # Semantic (no chunk_size)
    {"pipeline_name": "dense_semantic",         "splitter_type": "semantic", "chunk_size": 0, "chunk_overlap": 0, "retrieval_mode": "dense",  "use_reranker": False},
    {"pipeline_name": "dense_semantic_rerank",  "splitter_type": "semantic", "chunk_size": 0, "chunk_overlap": 0, "retrieval_mode": "dense",  "use_reranker": True},
    {"pipeline_name": "hybrid_semantic",        "splitter_type": "semantic", "chunk_size": 0, "chunk_overlap": 0, "retrieval_mode": "hybrid", "use_reranker": False},
    {"pipeline_name": "hybrid_semantic_rerank", "splitter_type": "semantic", "chunk_size": 0, "chunk_overlap": 0, "retrieval_mode": "hybrid", "use_reranker": True},
]

# === MAIN LOOP ===
metric_columns = [
    'context_precision', 'faithfulness', 'answer_relevancy',
    'context_recall', 'answer_correctness', 'answer_similarity'
]

completed = []
skipped = []
failed = []
total_start = time.time()
total_pipelines = len(evaluation_matrix)

for i, config in enumerate(evaluation_matrix):
    pipeline_name = config["pipeline_name"]

    # Skip if already done
    path = f"{RESULTS_DIR}/{pipeline_name}_final_results.csv"
    if os.path.exists(path):
        print(f"[SKIP] {pipeline_name}")
        skipped.append(pipeline_name)
        continue

    print(f"\n{'='*60}")
    print(f"[{i+1}/{total_pipelines}] {pipeline_name}")
    print(f"{'='*60}")
    pipeline_start = time.time()

    try:
        print(f"[1/3] Generating answers...")
        generated_answers = run_single_test(name, config, eval_questions, llm)
        if not generated_answers:
            failed.append(pipeline_name)
            cleanup_memory()
            continue

        df_results = pd.DataFrame(generated_answers)
        df_results["question_type"] = question_types
        df_results["chunk_size"] = config["chunk_size"]
        df_results["chunk_overlap"] = config["chunk_overlap"]
        gen_time = time.time() - pipeline_start
        print(f"[OK] {len(df_results)} answers in {gen_time:.0f}s ({gen_time/len(eval_questions):.1f}s/question)")

        # Save raw backup
        backup_path = f"{RESULTS_DIR}/{pipeline_name}_answers_raw.csv"
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
        final_df.to_csv(path, index=False)
        print(f"  -> {path}")

        total_time = time.time() - pipeline_start
        f = final_df['faithfulness'].mean()
        ar = final_df['answer_relevancy'].mean()
        cp = final_df['context_precision'].mean()
        cr = final_df['context_recall'].mean()
        print(f"  F={f:.3f} AR={ar:.3f} CP={cp:.3f} CR={cr:.3f} | {total_time:.0f}s total")
        completed.append(pipeline_name)

    except Exception as e:
        print(f"[ERROR] {pipeline_name}: {e}")
        import traceback
        traceback.print_exc()
        failed.append(pipeline_name)

    cleanup_memory()

# === SUMMARY ===
total_elapsed = time.time() - total_start
print(f"\n{'='*60}")
print(f"ALL DONE in {total_elapsed/3600:.1f}h")
print(f"  Completed: {len(completed)} | Skipped: {len(skipped)} | Failed: {len(failed)}")
if failed:
    print(f"  Failed: {failed}")
print(f"{'='*60}")
