"""
Post-hoc Re-Scoring von context_recall / faithfulness fuer bestehende Result-CSVs.

Hintergrund: Der RAGAS-Judge (gpt-3.5-turbo-16k) liefert seit ca. Mitte 2026
gelegentlich JSON-Formate, die der starre Parser von ragas 0.1.3 nicht versteht
(int statt str bei "verdict"/"Attributed", Dict-of-Dicts statt Liste). Das fuehrte
zu stillen NaN-Scores. Dieses Skript patcht die Parser zur Laufzeit (kein Eingriff
in site-packages) und bewertet betroffene Zeilen neu.

Usage (aus backend/, OPENAI_API_KEY in .env):
    # context_recall fuer ALLE Zeilen neu (einheitlicher Parser-Stand),
    # faithfulness nur fuer NaN-Zeilen:
    python rescore_ragas_nans.py results/Washington/llm_comparison_clean

    # mehrere Ordner:
    python rescore_ragas_nans.py results/Washington/final_run_42Q_kinit_fixed results/Eu/final_run_kinit_fixed
"""
import os, sys, ast, glob
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY fehlt (.env)"

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_recall, faithfulness
from ragas.metrics._context_recall import ContextRecall
from ragas.metrics._faithfulness import Faithfulness

# ---------------------------------------------------------------------------
# Runtime-Patches: drift-tolerante Parser (ragas 0.1.3)
# ---------------------------------------------------------------------------

def _unwrap(obj, nested_key=None):
    """{"classification": [...]} und {"statement_1": {...}, ...} -> Liste von Dicts."""
    if isinstance(obj, dict) and nested_key and nested_key in obj:
        obj = obj[nested_key]
    if isinstance(obj, dict) and obj and all(isinstance(v, dict) for v in obj.values()):
        obj = list(obj.values())
    obj = obj if isinstance(obj, list) else [obj]
    return [item if isinstance(item, dict) else {} for item in obj]


def _cr_compute_score(self, response):
    items = _unwrap(response, "classification")

    def verdict(item):
        val = None
        for key in ("Attributed", "attributed", "ATTRIBUTED", "Attribution"):
            if key in item:
                val = item[key]
                break
        if val is None:
            return np.nan
        s = str(val).strip().lower()
        if s in ("1", "yes", "true"):
            return 1
        if s in ("0", "no", "false"):
            return 0
        return np.nan

    scores = [verdict(i) for i in items]
    return sum(scores) / len(scores) if scores else np.nan


def _f_compute_score(self, output):
    vmap = {"1": 1, "0": 0, "null": np.nan, "-1": np.nan,
            "yes": 1, "no": 0, "true": 1, "false": 0}
    items = _unwrap(output)
    if not items:
        return np.nan
    total = sum(vmap.get(str(i.get("verdict", "")).strip().lower(), np.nan) for i in items)
    return total / len(items)


ContextRecall._compute_score = _cr_compute_score
Faithfulness._compute_score = _f_compute_score

# ---------------------------------------------------------------------------

GT_FILES = {
    "Washington": "./autogen_questions/Washington/hilfreich.csv",
    "Eu": "./autogen_questions/Eu/hilfreich2.csv",
}


def load_gt(corpus):
    g = pd.read_csv(GT_FILES[corpus])
    return dict(zip(g["question"].astype(str).str.strip(), g["ground_truth"].astype(str)))


def rescore(rows, metric, max_retries=4):
    scores = np.full(len(rows), np.nan)
    pending = np.arange(len(rows))
    for _ in range(max_retries):
        if len(pending) == 0:
            break
        sub = rows.iloc[pending]
        data = {"question": sub["question"].astype(str).tolist(),
                "contexts": sub["_contexts"].tolist()}
        if metric.name == "context_recall":
            data["ground_truth"] = sub["_gt"].astype(str).tolist()
        else:
            data["answer"] = sub["answer"].astype(str).tolist()
        res = evaluate(Dataset.from_dict(data), metrics=[metric], raise_exceptions=False)
        vals = res.to_pandas()[metric.name].values
        scores[pending] = vals
        pending = pending[np.isnan(vals)]
    return scores


if __name__ == "__main__":
    dirs = sys.argv[1:] or ["results"]
    files = sorted(f for d in dirs for f in glob.glob(os.path.join(d, "**", "*_final_results.csv"), recursive=True))
    print(f"{len(files)} CSVs")
    for f in files:
        corpus = "Eu" if f"{os.sep}Eu{os.sep}" in f or f.startswith(f"results{os.sep}Eu") or "/Eu/" in f.replace("\\", "/") else "Washington"
        gt = load_gt(corpus)
        df = pd.read_csv(f)
        df["_contexts"] = df["contexts"].apply(ast.literal_eval)
        df["_gt"] = df["question"].astype(str).str.strip().map(gt)

        mask = df["_gt"].notna()
        df.loc[mask, "context_recall"] = rescore(df[mask], context_recall)
        f_nan = df["faithfulness"].isna()
        if f_nan.sum():
            df.loc[f_nan, "faithfulness"] = rescore(df[f_nan], faithfulness)

        df.drop(columns=["_contexts", "_gt"]).to_csv(f, index=False)
        print(f"{f}:  CR={df['context_recall'].mean():.3f} (NaN {int(df['context_recall'].isna().sum())})  "
              f"F={df['faithfulness'].mean():.3f} (NaN {int(df['faithfulness'].isna().sum())})")
