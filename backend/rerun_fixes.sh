#!/bin/bash
# =============================================================================
# rerun_fixes.sh — stellt die von zwei Evaluations-Bugs betroffenen Läufe nach.
#
# Teil A (RQ2, LLM-Vergleich): Im Original-Run bekamen GPT-3.5 und Phi-3
#   Mistral-formatierte Prompts, und evaluiert wurde die Impersonation-Ausgabe
#   (bei GPT-3.5: nur der Eyewitness-Teil). Rerun: modellgerechte Prompts,
#   faktische Antwort wird direkt evaluiert (--no-impersonation), ALLE drei
#   Generatoren neu, damit die Zahlen vergleichbar sind.
#
# Teil B (RQ1c, dense+rerank): k_init wurde für dense Retriever nie gesetzt
#   (VectorStoreRetriever hat kein .k-Attribut) — der Reranker sah nur 3 statt
#   10 Kandidaten. Rerun der 6 dense_*_rerank-Pipelines mit gefixtem k_init.
#   Impersonation bleibt AN, damit die Ergebnisse mit den bestehenden
#   dense_*-Läufen (final_run) vergleichbar sind.
#
# Voraussetzungen: setup_gpu_vm.sh gelaufen, Modelle in ./models/,
#   OPENAI_API_KEY in .env (für GPT-3.5-Generator UND RAGAS-Judge).
# Laufzeit: ca. 2–3 h auf RTX 5090.
# =============================================================================
set -e
# OPENAI_API_KEY aus .env in die Umgebung laden (RAGAS-Judge braucht ihn in JEDEM Lauf)
if [ -f .env ]; then set -a; source .env; set +a; fi
export GPU_LAYERS=-1
export TOKENIZERS_PARALLELISM=false

echo "==============================================="
echo " Teil A: RQ2 LLM-Vergleich (clean, 3 Generatoren)"
echo "==============================================="
python run_evaluation.py --corpus Washington --pipelines dense_semantic \
    --llm mistral --no-impersonation --output-dir llm_comparison_clean

python run_evaluation.py --corpus Washington --pipelines dense_semantic \
    --llm gpt35 --no-impersonation --output-dir llm_comparison_clean

python run_evaluation.py --corpus Washington --pipelines dense_semantic \
    --llm phi3 --no-impersonation --output-dir llm_comparison_clean

echo "==============================================="
echo " Teil B: RQ1c dense+rerank mit k_init-Fix"
echo "==============================================="
python run_evaluation.py --corpus Washington \
    --pipelines dense_recursive dense_sentence dense_semantic dense_recursive_rerank dense_sentence_rerank dense_semantic_rerank \
    --output-dir final_run_42Q_kinit_fixed

python run_evaluation.py --corpus Eu \
    --questions-file ./autogen_questions/Eu/hilfreich2.csv \
    --pipelines dense_recursive dense_sentence dense_semantic dense_recursive_rerank dense_sentence_rerank dense_semantic_rerank \
    --output-dir final_run_kinit_fixed

echo ""
echo "==============================================="
echo " FERTIG. Ergebnisse:"
echo "   results/Washington/llm_comparison_clean/"
echo "   results/Washington/final_run_42Q_kinit_fixed/"
echo "   results/Eu/final_run_kinit_fixed/"
echo " Diese drei Ordner herunterladen (z.B. als zip):"
echo "   zip -r rerun_results.zip results/Washington/llm_comparison_clean results/Washington/final_run_42Q_kinit_fixed results/Eu/final_run_kinit_fixed"
echo "==============================================="
