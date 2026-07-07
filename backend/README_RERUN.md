# Rerun-Anleitung: Fixes für RQ2 (LLM-Vergleich) und RQ1c (dense+rerank)

## Was und warum

Beim Code-Review wurden zwei Bugs in der Evaluations-Pipeline gefunden:

1. **RQ2 / LLM-Vergleich**: GPT-3.5 und Phi-3 bekamen Mistral-formatierte Prompts
   (`<s>[INST] <<SYS>>`), und evaluiert wurde die Ausgabe der Impersonation-Stufe
   statt der faktischen Antwort (bei GPT-3.5 blieb dadurch nur der Eyewitness-Satz
   übrig). → Rerun aller 3 Generatoren mit modellgerechten Prompts und direkter
   Evaluation der faktischen Antwort (`--no-impersonation`).
2. **RQ1c / dense+rerank**: Der Kandidatenpool fürs Reranking wurde bei dense
   Retrievern nie auf 10 erhöht (LangChain `VectorStoreRetriever` hat kein
   `.k`-Attribut — der `hasattr`-Check schlug still fehl). Der Reranker hat bei
   den 6 `dense_*_rerank`-Läufen also nur 3 Dokumente umsortiert statt 10 → 3
   auszuwählen. → Rerun dieser 6 Pipelines mit dem Fix.

Die Fixes stecken in `packages/init_chain.py` (k_init via `search_kwargs`,
neue Parameter `impersonate` und `prompt_style`), `prompts/prompt_styles.py`
(neu) und `run_evaluation.py` (Flags `--no-impersonation`, `--prompt-style`,
`GPU_LAYERS`-Env-Var). Alles andere ist unverändert — bestehende Läufe bleiben
bit-identisch reproduzierbar (Default: `impersonate=True`, `prompt_style="mistral"`).

**Gesamtaufwand: ~2–3 h GPU-Zeit ≈ 1 € Vast.ai + ~2–3 € OpenAI-API (RAGAS-Judge).**

---

## Schritt 1: GPU mieten (Vast.ai)

1. https://cloud.vast.ai → Console → **Search** (Templates: „PyTorch (cuDNN Devel)" o. ä.)
2. Filter: **RTX 5090** (oder 4090), **Disk ≥ 60 GB**, verlässlicher Host (Reliability > 99 %)
3. Preis war beim letzten Mal ~0,29 $/h für eine 5090
4. **Rent** → Instanz startet → über **Open SSH** oder das Jupyter-Terminal verbinden

> Tipp aus dem letzten Run: Das Web-Terminal von Vast.ai zerhackt lange Befehle.
> Entweder per echtem SSH verbinden oder Befehle in Dateien schreiben (`nano`).

## Schritt 2: Code hochladen

Das Zip (`rerun_package.zip`) hochladen — am einfachsten per Jupyter-Upload
(Dateibrowser → Upload) oder per scp:

```bash
scp -P <PORT> rerun_package.zip root@<HOST>:/workspace/
```

Dann auf der VM:

```bash
cd /workspace
unzip rerun_package.zip
cd backend
```

## Schritt 3: Setup (einmalig, ~15–20 min)

```bash
chmod +x setup_gpu_vm.sh
./setup_gpu_vm.sh
```

Danach die bekannten Stolpersteine vom letzten Run fixen:

```bash
# RTX 5090 (Blackwell/sm_120) braucht neueres llama-cpp-python mit CUDA-Build:
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python==0.3.16 --no-cache-dir --force-reinstall

# RAGAS braucht älteres httpx (sonst "proxies"-Fehler):
pip install "httpx<0.28"

# numpy 2.x bricht langchain:
pip install "numpy<2"

# pandas 3 bricht die RAGAS-Validierung ("question should be of type string"):
pip install "pandas<3"
```

## Schritt 4: Modelle herunterladen (~7 GB)

```bash
mkdir -p models
wget -O models/mistral-7b-instruct-v0.1.Q4_K_M.gguf \
  "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
wget -O models/Phi-3-mini-4k-instruct-q4.gguf \
  "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf"
```

## Schritt 5: OpenAI-Key setzen

Wird für den GPT-3.5-Generator **und** den RAGAS-Judge gebraucht:

```bash
echo 'OPENAI_API_KEY=sk-...' > .env
```

## Schritt 6: Smoke-Test (~5 min)

Erst mit 3 Fragen prüfen, dass alles läuft:

```bash
export GPU_LAYERS=-1
python run_evaluation.py --corpus Washington --pipelines dense_semantic \
  --llm gpt35 --no-impersonation --output-dir smoke_test --limit 3
```

Checkliste im Output:
- `[CONFIG] N_GPU_LAYERS = -1`
- `[CONFIG] prompt_style=plain, impersonation=OFF`
- `[INFO] Impersonation disabled — evaluating factual answer directly.`
- Die Antworten in `results/Washington/smoke_test/*.csv` sind echte faktische
  Antworten (nicht „Eyewitness: …")

## Schritt 7: Vollständiger Rerun (~2–3 h)

```bash
chmod +x rerun_fixes.sh
nohup bash rerun_fixes.sh > rerun.log 2>&1 &
tail -f rerun.log
```

Beim Teil-B-Lauf prüfen: `num_initial_docs` muss jetzt **10** sein (vorher 3).

## Schritt 8: Ergebnisse herunterladen, Instanz zerstören

```bash
zip -r rerun_results.zip \
  results/Washington/llm_comparison_clean \
  results/Washington/final_run_42Q_kinit_fixed \
  results/Eu/final_run_kinit_fixed
```

Zip per Jupyter-Dateibrowser (oder scp) herunterladen, dann Instanz auf
Vast.ai **destroyen** (nicht nur stoppen — Stoppen kostet weiter Disk-Miete).

---

## Post-hoc CR-Re-Scoring (Judge-JSON-Drift)

Der Judge (gpt-3.5-turbo-16k) liefert seit Mitte 2026 teils JSON-Formate, die
ragas 0.1.3 nicht parst (int-Verdicts, Dict-of-Dicts) → stille NaN bei
context_recall/faithfulness. `rescore_ragas_nans.py` patcht die Parser zur
Laufzeit und bewertet betroffene Zeilen nach (context_recall komplett neu für
einheitlichen Parser-Stand, faithfulness nur NaN-Zeilen):

```bash
python rescore_ragas_nans.py results/Washington/llm_comparison_clean \
    results/Washington/final_run_42Q_kinit_fixed results/Eu/final_run_kinit_fixed
```

Die eingecheckten `*_final_results.csv` in diesen drei Ordnern sind bereits
nachbewertet (alle CR-NaNs aufgelöst; verbleibende F-NaNs sind "I don't
know"-Antworten ohne prüfbare Aussagen und bleiben per Design NaN).

## Auswertung danach

- **RQ2**: Mittelwerte (F, AR, CP, CR) der 3 Läufe in `llm_comparison_clean/`
  vergleichen. Diese Zahlen ersetzen die alten aus `llm_comparison/`.
- **RQ1c**: `dense_*_rerank` (neu, k_init=10) gegen die bestehenden `dense_*`
  (final_run, ohne Rerank) paaren — Wilcoxon + Holm wie gehabt. Die alten
  `dense_*_rerank`-Läufe sind obsolet. Die hybrid-Läufe waren nicht betroffen
  und bleiben gültig.
