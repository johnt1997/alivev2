j#!/usr/bin/env python3
"""
Vollständiges Evaluations-Script für alle RAG-Pipelines.

Unterstützt:
- Alle 12 Pipeline-Kombinationen (3 Chunker × 4 Modi)
- LLM-Vergleich (Mistral, GPT-3.5, Phi-3, etc.)
- Chunk-Size Experimente (500, 1000, 1500, etc.)

Beispiel-Aufrufe:
    # Alle 12 Standard-Pipelines
    python run_evaluation.py --corpus Washington

    # Nur bestimmte Pipelines
    python run_evaluation.py --corpus Washington --pipelines dense_semantic hybrid_semantic_rerank

    # LLM-Vergleich auf dense_semantic
    python run_evaluation.py --corpus Washington --pipelines dense_semantic --llm gpt35 --output-dir llm_comparison

    # Chunk-Size Experiment
    python run_evaluation.py --corpus Washington --pipelines dense_recursive --chunk-sizes 500 1000 1500 --output-dir chunk_size_experiment
"""

import os
import sys
import argparse
import gc
import json
import subprocess
import pandas as pd
from tqdm import tqdm

# Umgebungsvariablen setzen BEVOR torch importiert wird
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    import torch
    HAS_TORCH = True
except (ImportError, OSError):
    HAS_TORCH = False
    print("[WARN] torch not available - GPU memory cleanup disabled")

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    answer_correctness,
    answer_similarity,
)

from packages.person import Person
from packages.globals import EMBEDDINGS
from packages.llm_config import LLMConfig
from packages.evaluate_rag import EvaluationPipeline
from packages.vector_store_handler import VectorStoreHandler, HybridRetriever
from packages.document_processing import DocumentProcessing
from packages.bm25_retriever import BM25Retriever
from packages.init_chain import InitializeQuesionAnsweringChain


# =============================================================================
# BM25 INDEX ERSTELLUNG
# =============================================================================

def build_bm25_index(corpus_name: str, splitter_type: str, chunk_size: int, chunk_overlap: int):
    """
    Erstellt BM25-Index für Hybrid-Retrieval.

    1. Chunked Dokumente -> JSONL
    2. JSONL -> Lucene Index (via pyserini)
    """
    jsonl_dir = f"./bm25_jsonl/{corpus_name}_{splitter_type}"
    index_dir = f"./bm25_indexes/{corpus_name}_{splitter_type}"

    # Schritt 1: JSONL erstellen
    print(f"[BM25] Creating JSONL for {splitter_type}...")
    os.makedirs(jsonl_dir, exist_ok=True)

    dp = DocumentProcessing(embeddings=EMBEDDINGS)
    chunks = dp.get_chunked_documents(
        directory_path=f"./data/{corpus_name}",
        splitter_type=splitter_type,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    jsonl_path = os.path.join(jsonl_dir, "docs.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            record = {
                "id": chunk.metadata.get("source", "unknown"),
                "contents": chunk.page_content
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[BM25] Wrote {len(chunks)} chunks to {jsonl_path}")

    # Schritt 2: Lucene Index bauen
    print(f"[BM25] Building Lucene index...")
    os.makedirs(index_dir, exist_ok=True)

    cmd = [
        sys.executable, "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", jsonl_dir,
        "--index", index_dir,
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", "4",
        "--storePositions", "--storeDocvectors", "--storeRaw"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[BM25] STDERR: {result.stderr}")
        raise RuntimeError(f"Failed to build BM25 index: {result.stderr}")

    # Validieren
    from pyserini.search.lucene import LuceneSearcher
    searcher = LuceneSearcher(index_dir)
    print(f"[BM25] Index ready: {index_dir} | num_docs={searcher.num_docs}")

    return index_dir


def ensure_bm25_index(corpus_name: str, splitter_type: str, chunk_size: int, chunk_overlap: int) -> str:
    """
    Stellt sicher dass BM25-Index existiert, erstellt ihn falls nötig.
    Gibt den Index-Pfad zurück.
    """
    index_dir = f"./bm25_indexes/{corpus_name}_{splitter_type}"

    if os.path.isdir(index_dir) and os.listdir(index_dir):
        print(f"[BM25] Using existing index: {index_dir}")
        return index_dir

    print(f"[BM25] Index not found, building: {index_dir}")
    return build_bm25_index(corpus_name, splitter_type, chunk_size, chunk_overlap)


# =============================================================================
# PIPELINE DEFINITIONEN
# =============================================================================

STANDARD_PIPELINES = {
    # Dense Retrieval
    "dense_recursive": {
        "splitter_type": "recursive", "chunk_size": 1000, "chunk_overlap": 30,
        "retrieval_mode": "dense", "use_reranker": False
    },
    "dense_sentence": {
        "splitter_type": "sentence_transformer", "chunk_size": 256, "chunk_overlap": 30,
        "retrieval_mode": "dense", "use_reranker": False
    },
    "dense_semantic": {
        "splitter_type": "semantic", "chunk_size": 0, "chunk_overlap": 0,
        "retrieval_mode": "dense", "use_reranker": False
    },
    # Dense + Reranker
    "dense_recursive_rerank": {
        "splitter_type": "recursive", "chunk_size": 1000, "chunk_overlap": 30,
        "retrieval_mode": "dense", "use_reranker": True
    },
    "dense_sentence_rerank": {
        "splitter_type": "sentence_transformer", "chunk_size": 256, "chunk_overlap": 30,
        "retrieval_mode": "dense", "use_reranker": True
    },
    "dense_semantic_rerank": {
        "splitter_type": "semantic", "chunk_size": 0, "chunk_overlap": 0,
        "retrieval_mode": "dense", "use_reranker": True
    },
    # Hybrid Retrieval
    "hybrid_recursive": {
        "splitter_type": "recursive", "chunk_size": 1000, "chunk_overlap": 30,
        "retrieval_mode": "hybrid", "use_reranker": False
    },
    "hybrid_sentence": {
        "splitter_type": "sentence_transformer", "chunk_size": 256, "chunk_overlap": 30,
        "retrieval_mode": "hybrid", "use_reranker": False
    },
    "hybrid_semantic": {
        "splitter_type": "semantic", "chunk_size": 0, "chunk_overlap": 0,
        "retrieval_mode": "hybrid", "use_reranker": False
    },
    # Hybrid + Reranker
    "hybrid_recursive_rerank": {
        "splitter_type": "recursive", "chunk_size": 1000, "chunk_overlap": 30,
        "retrieval_mode": "hybrid", "use_reranker": True
    },
    "hybrid_sentence_rerank": {
        "splitter_type": "sentence_transformer", "chunk_size": 256, "chunk_overlap": 30,
        "retrieval_mode": "hybrid", "use_reranker": True
    },
    "hybrid_semantic_rerank": {
        "splitter_type": "semantic", "chunk_size": 0, "chunk_overlap": 0,
        "retrieval_mode": "hybrid", "use_reranker": True
    },
}

RAGAS_METRICS = [
    context_precision,
    faithfulness,
    answer_relevancy,
    context_recall,
    answer_correctness,
    answer_similarity,
]

METRIC_COLUMNS = [
    'context_precision', 'faithfulness', 'answer_relevancy',
    'context_recall', 'answer_correctness', 'semantic_similarity'
]


# =============================================================================
# HILFSFUNKTIONEN
# =============================================================================

def cleanup_memory():
    """Gibt GPU/MPS Memory frei."""
    gc.collect()
    if HAS_TORCH:
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()


def get_llm(llm_type: str, model_path: str = None):
    """
    Initialisiert das LLM basierend auf dem Typ.

    Args:
        llm_type: "mistral" | "gpt35" | "phi3" | "custom"
        model_path: Pfad zum GGUF-Modell (für custom)
    """
    llm_config = LLMConfig(temperature=0.0)

    if llm_type == "mistral":
        return llm_config.get_local_llm(use_openai=False)
    elif llm_type == "gpt35":
        return llm_config.get_local_llm(use_openai=True)
    elif llm_type == "phi3":
        # Phi-3 mit speziellem Pfad
        if model_path is None:
            model_path = "./models/Phi-3-mini-4k-instruct-q4.gguf"
        from langchain.llms import LlamaCpp
        return LlamaCpp(
            model_path=model_path,
            n_ctx=4096,
            n_gpu_layers=-1,
            temperature=0.0,
            verbose=False,
        )
    elif llm_type == "custom" and model_path:
        from langchain.llms import LlamaCpp
        return LlamaCpp(
            model_path=model_path,
            n_ctx=4096,
            n_gpu_layers=-1,
            temperature=0.0,
            verbose=False,
        )
    else:
        raise ValueError(f"Unknown LLM type: {llm_type}")


def run_single_pipeline(
    corpus_name: str,
    config: dict,
    questions: list,
    llm,
    search_kwargs_num: int = 3,
) -> list:
    """
    Führt eine einzelne Pipeline aus und gibt die Antworten zurück.
    """
    person = Person(name=corpus_name)
    embedding = EMBEDDINGS

    splitter_type = config["splitter_type"]
    chunk_size = config["chunk_size"]
    chunk_overlap = config["chunk_overlap"]
    retrieval_mode = config["retrieval_mode"]
    use_reranker = config["use_reranker"]

    # Vector Store laden oder erstellen
    vectorstore_handler = VectorStoreHandler(
        embeddings=embedding,
        search_kwargs_num=search_kwargs_num
    )

    database_path = vectorstore_handler._get_vector_store_path(
        vector_store_name=person.name,
        splitter_type=splitter_type,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    if not os.path.exists(database_path):
        print(f"[INFO] Creating database at {database_path}...")
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
        print(f"[INFO] Loading existing database from {database_path}...")
        _, db = vectorstore_handler.get_db_and_retriever(
            vector_store_name=person.name,
            splitter_type=splitter_type,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    if db is None:
        raise RuntimeError(f"Failed to load/create database for {splitter_type}")

    # Retriever erstellen
    if retrieval_mode == 'hybrid':
        print("[INFO] Using HYBRID retriever")
        # BM25-Index automatisch erstellen falls nicht vorhanden
        bm25_index_dir = ensure_bm25_index(
            corpus_name=corpus_name,
            splitter_type=splitter_type,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        bm25_retriever = BM25Retriever(bm25_index_dir)
        retriever = HybridRetriever(db=db, bm25_retriever=bm25_retriever, k=search_kwargs_num)
    else:
        print("[INFO] Using DENSE retriever")
        retriever = db.as_retriever(search_kwargs={"k": search_kwargs_num})

    # QA Chain erstellen
    qa_chain = InitializeQuesionAnsweringChain(
        llm=llm,
        retriever=retriever,
        db=db,
        person=person,
        search_kwargs_num=search_kwargs_num,
        use_reranker=use_reranker
    )

    # Antworten generieren
    eval_pipeline = EvaluationPipeline(
        qa_chain=qa_chain,
        eval_questions=questions
    )

    return eval_pipeline.generate_answers_with_metadata()


def run_ragas_evaluation(df_generated: pd.DataFrame, ground_truths: list) -> pd.DataFrame:
    """Führt RAGAS Evaluation durch und gibt DataFrame mit Scores zurück."""
    df_generated = df_generated.copy()
    df_generated['ground_truth'] = ground_truths
    df_generated['ground_truths'] = df_generated['ground_truth'].apply(lambda x: [x])

    ragas_dataset = Dataset.from_pandas(df_generated)
    ragas_results = evaluate(ragas_dataset, metrics=RAGAS_METRICS)

    df_ragas_scores = ragas_results.to_pandas()
    df_scores_only = df_ragas_scores[METRIC_COLUMNS]

    final_df = pd.concat([
        df_generated.reset_index(drop=True),
        df_scores_only.reset_index(drop=True)
    ], axis=1)

    if 'ground_truth' in final_df.columns:
        final_df = final_df.drop(columns=['ground_truth'])

    return final_df


# =============================================================================
# HAUPTFUNKTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="RAG Pipeline Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument("--corpus", required=True, help="Name des Corpus (z.B. Washington)")
    parser.add_argument("--pipelines", nargs="+", default=list(STANDARD_PIPELINES.keys()),
                        help="Welche Pipelines ausführen (default: alle 12)")
    parser.add_argument("--llm", default="mistral", choices=["mistral", "gpt35", "phi3", "custom"],
                        help="Welches LLM verwenden")
    parser.add_argument("--model-path", help="Pfad zum GGUF-Modell (für --llm custom)")
    parser.add_argument("--chunk-sizes", nargs="+", type=int,
                        help="Chunk-Size Experiment (z.B. 500 1000 1500)")
    parser.add_argument("--output-dir", default="final_run",
                        help="Unterordner für Results (default: final_run)")
    parser.add_argument("--questions-file", help="CSV mit Fragen (default: autogen_questions/{corpus}/hilfreich.csv)")
    parser.add_argument("--limit", type=int, help="Nur erste N Fragen (für Tests)")

    args = parser.parse_args()

    # Fragen laden
    questions_file = args.questions_file or f"./autogen_questions/{args.corpus}/hilfreich.csv"
    if not os.path.exists(questions_file):
        print(f"[ERROR] Questions file not found: {questions_file}")
        sys.exit(1)

    df_eval = pd.read_csv(questions_file)
    if args.limit:
        df_eval = df_eval.head(args.limit)

    eval_questions = df_eval["question"].tolist()
    question_types = df_eval["question_type"].tolist() if "question_type" in df_eval.columns else [None] * len(eval_questions)
    ground_truths = df_eval["ground_truth"].tolist()

    print(f"[INFO] Loaded {len(eval_questions)} questions from {questions_file}")

    # LLM initialisieren
    print(f"[INFO] Loading LLM: {args.llm}")
    llm = get_llm(args.llm, args.model_path)

    # Output-Verzeichnis
    output_dir = f"./results/{args.corpus}/{args.output_dir}"
    os.makedirs(output_dir, exist_ok=True)

    # Pipelines ausführen
    successful = 0
    failed = 0

    for pipeline_name in args.pipelines:
        if pipeline_name not in STANDARD_PIPELINES:
            print(f"[WARN] Unknown pipeline: {pipeline_name}, skipping")
            continue

        base_config = STANDARD_PIPELINES[pipeline_name].copy()

        # Chunk-Size Experiment?
        if args.chunk_sizes and base_config["splitter_type"] == "recursive":
            chunk_sizes = args.chunk_sizes
        else:
            chunk_sizes = [base_config["chunk_size"]]

        for chunk_size in chunk_sizes:
            config = base_config.copy()
            config["chunk_size"] = chunk_size

            # Output-Name
            if len(chunk_sizes) > 1:
                output_name = f"{pipeline_name}_{chunk_size}"
            elif args.llm != "mistral":
                output_name = f"{pipeline_name}_{args.llm}"
            else:
                output_name = pipeline_name

            print(f"\n{'='*60}")
            print(f"RUNNING: {output_name}")
            print(f"{'='*60}")

            try:
                # Antworten generieren
                generated_answers = run_single_pipeline(
                    corpus_name=args.corpus,
                    config=config,
                    questions=eval_questions,
                    llm=llm,
                )

                df_results = pd.DataFrame(generated_answers)
                df_results["question_type"] = question_types

                # Raw Answers speichern
                raw_file = f"{output_dir}/{output_name}_answers_raw.csv"
                df_results.to_csv(raw_file, index=False)
                print(f"[INFO] Saved raw answers: {raw_file}")

                # RAGAS Evaluation
                print("[INFO] Running RAGAS evaluation...")
                final_df = run_ragas_evaluation(df_results, ground_truths)

                # Final Results speichern
                final_file = f"{output_dir}/{output_name}_final_results.csv"
                final_df.to_csv(final_file, index=False)
                print(f"[INFO] Saved final results: {final_file}")

                # Metrics ausgeben
                print(f"\n[RESULTS] {output_name}:")
                for metric in METRIC_COLUMNS:
                    if metric in final_df.columns:
                        print(f"  {metric}: {final_df[metric].mean():.3f}")

                successful += 1

            except Exception as e:
                print(f"[ERROR] Pipeline {output_name} failed: {e}")
                import traceback
                traceback.print_exc()
                failed += 1

            finally:
                cleanup_memory()

    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETE")
    print(f"Successful: {successful}, Failed: {failed}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
