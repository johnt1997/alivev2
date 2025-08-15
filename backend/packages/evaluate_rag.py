# evaluate_rag.py (Vervollständigtes Skeleton)

import json
import time
import pandas as pd
import os
from tqdm import tqdm
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    faithfulness, 
    answer_relevancy, 
    answer_similarity,
    context_precision, 
    context_recall, 
    answer_correctness
)

# Importiere deine Klassen
from packages.init_chain import InitializeQuesionAnsweringChain
from packages.person import Person

class EvaluationPipeline:
    """
    Diese Klasse kapselt den Prozess der Antwortgenerierung für die Evaluation.
    Sie nimmt eine fertig konfigurierte QA-Kette entgegen.
    """
    def __init__(self, qa_chain: InitializeQuesionAnsweringChain, eval_questions: list[str]):
        self.qa_chain = qa_chain
        self.eval_questions = eval_questions

    def generate_answers_with_metadata(self, eval_questions = None):
        """
        Führt die QA-Kette für alle Fragen aus und sammelt Antworten sowie Metadaten.
        """
        if eval_questions is None:
            eval_questions = self.eval_questions
        results = []
        for question in tqdm(self.eval_questions, desc="Generating Answers"):
            
            start_time = time.time()
            # Nutze die neue, saubere 'answer'-Methode
            answer, docs_with_scores, meta = self.qa_chain.answer(question)
            end_time = time.time()
            
            # Extrahiere den internen Factual-Score
            # HINWEIS: Hierfür muss die 'answer'-Methode angepasst werden,
            # um auch das 'dict_query' zurückzugeben
            # factuality_score = ... 

            results.append({
                "question": question,
                "answer": answer,
                "contexts": [d.page_content for d, _ in docs_with_scores],
                "used_reranker": meta.get("used_reranker", None),
                "k_init": meta.get("k_init", None),
                "top_n": meta.get("top_n", None),
                "num_initial_docs": meta.get("num_initial_docs", None),
                "num_final_docs": meta.get("num_final_docs", None),
                "runtime_seconds": end_time - start_time,
                "factuality_score": meta["factuality_score"],
                
            })
        return results

def run_evaluation_scenario(
    name: str, 
    retriever, 
    llm, 
    db, 
    use_reranker: bool, 
    config: dict, 
    questions_df: pd.DataFrame, 
    output_dir: str
):
    """
    Führt einen kompletten Evaluations-Durchlauf für ein Szenario durch.
    """
    print(f"--- Starting Evaluation: {config.get('pipeline_name', 'Unnamed')} ---")
    
    # 1. QA-Kette für dieses Szenario initialisieren
    qa_chain = InitializeQuesionAnsweringChain(
        llm=llm, 
        retriever=retriever, 
        db=db, 
        person=Person(name),
        search_kwargs_num=config.get("top_k", 3), 
        use_reranker=use_reranker
    )
    
    # 2. Pipeline zur Antwortgenerierung erstellen
    pipeline = EvaluationPipeline(qa_chain, questions_df["user_input"].tolist())
    generated_answers = pipeline.generate_answers_with_metadata()
    
    # 3. RAGAS-Dataset vorbereiten
    ragas_dataset_list = []
    for i, entry in enumerate(generated_answers):
        ragas_dataset_list.append({
            "question": entry["question"],
            "answer": entry["answer"],
            "contexts": entry["contexts"],
            "ground_truths": [questions_df.loc[i, "reference"]],  # LISTE!
        })
    
    ragas_dataset = Dataset.from_list(ragas_dataset_list)

    # 4. RAGAS Evaluation durchführen
    print("Running RAGAS evaluation...")
    ragas_results = evaluate(
        ragas_dataset, 
        metrics=[
            context_precision, 
            context_recall, 
            faithfulness, 
            answer_relevancy, 
            answer_similarity, 
            answer_correctness
        ]
    )
    
    # 5. Alle Ergebnisse zusammenführen und speichern
    df_answers = pd.DataFrame(generated_answers)
    df_ragas = ragas_results.to_pandas()
    
    final_results = pd.concat([df_answers, df_ragas.drop(columns=['question', 'answer', 'contexts', 'ground_truth'])], axis=1)

    # Eindeutige ID für diesen Lauf erstellen
    ts = int(time.time())
    run_id = f'{config["pipeline_name"]}_model-{config["model_name"]}_rerank-{int(use_reranker)}_{ts}'
    
    # Ergebnisse und Konfiguration speichern
    os.makedirs(output_dir, exist_ok=True)
    final_results.to_csv(f"{output_dir}/{run_id}_results.csv", index=False)
    with open(f"{output_dir}/{run_id}_config.json", "w") as f:
        # Konfiguration speichern (DB-Objekt kann nicht direkt gespeichert werden, daher entfernen)
        config_to_save = config.copy()
        config_to_save.pop('db', None) 
        json.dump(config_to_save, f, indent=2)
        
    print(f"--- Evaluation finished. Results saved with ID: {run_id} ---")
    return final_results