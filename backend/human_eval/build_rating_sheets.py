"""
Baut den Bewertungsbogen fuer den Human-Evaluation-Pilot (Faithfulness).

Stichprobe: 30 Antworten, je 10 aus drei Quellen (Washington-Pipelines,
EU-Pipelines, Generator-Vergleich), innerhalb jeder Quelle stratifiziert nach
RAGAS-Faithfulness (4 niedrig / 3 mittel / 3 hoch), damit der Score-Bereich
abgedeckt ist. Jede Frage kommt hoechstens einmal vor; Verweigerungen
("I don't know"), fehlende Scores und sehr lange Kontexte (>6000 Zeichen,
Lesaufwand) sind ausgeschlossen.

Ausgabe (in diesem Ordner):
  Rater_A.xlsx, Rater_B.xlsx   identische Boegen, einer pro Rater
  key_DO_NOT_SHARE.csv          Zuordnung Eintrag -> Quelle, Pipeline, RAGAS-Score,
                                Zelle der Bewertung (nur fuer die Auswertung)

Aufruf (aus backend/):  python human_eval/build_rating_sheets.py
"""
import ast
import glob
import os
import re

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.datavalidation import DataValidation

SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
MAX_CTX = 6000
BINS = [("low", 0.0, 0.6, 4), ("mid", 0.6, 1.0, 3), ("high", 1.0, 1.01, 3)]

SOURCES = {
    "WA": os.path.join(RESULTS, "Washington", "final_run_42Q_kinit_fixed", "*_final_results.csv"),
    "EU": os.path.join(RESULTS, "Eu", "final_run_kinit_fixed", "*_final_results.csv"),
    "LLM": os.path.join(RESULTS, "Washington", "llm_comparison_clean", "*_final_results.csv"),
}


def is_refusal(answer: str) -> bool:
    s = str(answer).strip().lower()
    return s.startswith(("i don't know", "i do not know", "i don’t know"))


def clean_answer(answer: str) -> str:
    # Washington-Laeufe mit Impersonation-Stufe beginnen mit einem Label-Echo
    return re.sub(r"^\s*Original Answer:\s*", "", str(answer)).strip()


def load_pool():
    frames = []
    for source, pattern in SOURCES.items():
        for f in sorted(glob.glob(pattern)):
            df = pd.read_csv(f)
            df["source"] = source
            df["file"] = os.path.relpath(f, RESULTS)
            df["pipeline"] = os.path.basename(f).replace("_final_results.csv", "")
            df["row"] = df.index
            df["_ctx"] = df["contexts"].apply(ast.literal_eval)
            df["_ctx_len"] = df["_ctx"].apply(lambda c: sum(len(x) for x in c))
            frames.append(df)
    pool = pd.concat(frames, ignore_index=True)
    pool = pool[pool["faithfulness"].notna()
                & ~pool["answer"].apply(is_refusal)
                & (pool["_ctx_len"] <= MAX_CTX)
                & (pool["_ctx"].apply(len) > 0)]
    return pool


def sample(pool: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    picked, used_questions = [], set()
    models = sorted(pool.loc[pool["source"] == "LLM", "pipeline"].unique())
    turn = 0  # Im Generator-Vergleich reihum ueber die Modelle, ueber alle Strata hinweg

    def take(cand, model=None):
        for _, r in cand.iterrows():
            q = str(r["question"]).strip()
            if q in used_questions or (model is not None and r["pipeline"] != model):
                continue
            used_questions.add(q)
            return r
        return None

    for source in SOURCES:
        sub = pool[pool["source"] == source]
        for label, lo, hi, n in BINS:
            cand = sub[(sub["faithfulness"] >= lo) & (sub["faithfulness"] < hi)]
            cand = cand.iloc[rng.permutation(len(cand))]
            for _ in range(n):
                r = None
                if source == "LLM":
                    for k in range(len(models)):
                        r = take(cand, models[(turn + k) % len(models)])
                        if r is not None:
                            turn += k + 1
                            break
                else:
                    r = take(cand)
                if r is None:
                    raise RuntimeError(f"Zu wenige Kandidaten: {source}/{label}")
                picked.append({**r.to_dict(), "stratum": label})
    items = pd.DataFrame(picked)
    items = items.iloc[rng.permutation(len(items))].reset_index(drop=True)
    items["item_id"] = [f"H{i + 1:02d}" for i in range(len(items))]
    return items


# ---------------------------------------------------------------------------
# Excel
# ---------------------------------------------------------------------------
FONT = "Arial"
F_BODY = Font(name=FONT, size=10)
F_BOLD = Font(name=FONT, size=10, bold=True)
F_HEAD = Font(name=FONT, size=12, bold=True, color="FFFFFF")
F_TITLE = Font(name=FONT, size=14, bold=True)
FILL_HEAD = PatternFill("solid", fgColor="1F4E78")
FILL_LABEL = PatternFill("solid", fgColor="DDEBF7")
FILL_INPUT = PatternFill("solid", fgColor="FFFF00")
FILL_EXAMPLE = PatternFill("solid", fgColor="EDEDED")
WRAP = Alignment(wrap_text=True, vertical="top")
THIN = Side(style="thin", color="A6A6A6")
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
CHARS_PER_LINE = 125  # bei Spaltenbreite 120


def row_height(text: str) -> float:
    lines = sum(max(1, -(-len(part) // CHARS_PER_LINE)) for part in str(text).split("\n"))
    return min(409, max(15, 13.5 * lines + 4))


CODEBOOK = [
    ("title", "Codebook: Bewertung der Quellentreue (Faithfulness)"),
    ("h", "Worum geht es?"),
    ("p", "Ein Chatbot beantwortet Fragen zu historischen Themen. Vor jeder Antwort sucht er drei "
          "Textstellen aus einer Dokumentsammlung heraus und soll seine Antwort ausschließlich auf "
          "diese Textstellen stützen. Deine Aufgabe: Beurteile für 30 Antworten, wie viel des "
          "Inhalts tatsächlich durch die jeweiligen Textstellen belegt ist."),
    ("h", "Die wichtigste Regel"),
    ("p", "Urteile NUR anhand der drei Textstellen – nicht anhand deines eigenen Wissens. Eine "
          "Aussage, die historisch korrekt ist, aber in keiner Textstelle steht, gilt als NICHT "
          "belegt. Umgekehrt zählt eine Aussage als belegt, wenn sie aus den Textstellen folgt, "
          "auch wenn du sie für falsch hältst."),
    ("h", "Vorgehen pro Eintrag"),
    ("p", "1. Lies die Antwort des Systems und zerlege sie gedanklich in einzelne Sachaussagen "
          "(meist 2–6).\n"
          "2. Prüfe jede Sachaussage: Steht sie so (oder sinngemäß) in mindestens einer Textstelle?\n"
          "3. Schätze den Anteil der belegten Aussagen und wähle den passenden Wert auf der Skala.\n"
          "4. Trage den Wert in die gelbe Zelle „Bewertung (1–5)“ ein. Wenn du dir unsicher bist, "
          "setze „Unsicher?“ auf „ja“ und schreib optional einen Satz in „Kommentar“."),
    ("h", "Skala"),
    ("scale", [("5", "vollständig belegt", "81–100 % der Aussagen stehen in den Textstellen"),
               ("4", "überwiegend belegt", "61–80 %: im Kern belegt, kleinere Zusätze"),
               ("3", "etwa zur Hälfte belegt", "41–60 %"),
               ("2", "überwiegend nicht belegt", "21–40 %: nur einzelne Aussagen belegt"),
               ("1", "nicht belegt / erfunden", "0–20 %: Inhalt stammt nicht aus den Textstellen")]),
    ("h", "Sonderfälle"),
    ("p", "• Umformulierungen, Zusammenfassungen und Schlussfolgerungen, die direkt aus den "
          "Textstellen folgen, zählen als belegt.\n"
          "• Ich-Perspektive oder Rollenspiel-Stil („Ich, George Washington …“) ist egal – bewertet "
          "wird nur der Sachinhalt.\n"
          "• Reine Floskeln ohne Sachgehalt („Das ist eine interessante Frage“) ignorieren.\n"
          "• Aussagen, die den Textstellen widersprechen, zählen wie nicht belegte Aussagen.\n"
          "• Die Textstellen wurden automatisch aus PDFs extrahiert und enthalten teils "
          "Zeilenumbrüche mitten im Satz, doppelte Wörter oder Folien-Fragmente. Das ist normal "
          "und kein Fehler der Antwort – lies großzügig über solche Artefakte hinweg.\n"
          "• Die Antworten und Textstellen sind auf Englisch; das Codebook ist auf Deutsch. "
          "Kommentare kannst du in beiden Sprachen schreiben."),
    ("h", "Beispiel (nicht Teil der Bewertung)"),
    ("example", {
        "Frage": "When was George Washington born and where?",
        "Textstelle 1": "George Washington was born on February 22, 1732, at his father's plantation "
                        "on Popes Creek in Westmoreland County, Virginia.",
        "Antwort": "George Washington was born on February 22, 1732, in Westmoreland County, "
                   "Virginia. He was the eldest son of Augustine Washington and Mary Ball, and he "
                   "later studied at the College of William & Mary.",
        "Bewertung": "3",
        "Begründung": "Drei Sachaussagen: Geburtsdatum (belegt), Geburtsort (belegt), Eltern und "
                      "Studium (nicht in der Textstelle; das Studium ist zudem historisch falsch). "
                      "Rund die Hälfte bis zwei Drittel belegt → 3, mit Tendenz zu 4. Bei solchen "
                      "Grenzfällen: den näheren Wert wählen, ggf. „Unsicher? = ja“.",
    }),
    ("h", "Organisatorisches"),
    ("p", "• Arbeite allein und sprich dich nicht mit der zweiten bewertenden Person ab – wir messen "
          "gerade, wie gut zwei unabhängige Urteile übereinstimmen.\n"
          "• Du siehst weder, welches System die Antwort erzeugt hat, noch welchen Score die "
          "automatische Bewertung vergeben hat. Das ist Absicht.\n"
          "• Aufwand: etwa 2–4 Minuten pro Eintrag, insgesamt 1,5–2 Stunden. Pausen sind "
          "ausdrücklich erlaubt; der Bogen speichert den Zwischenstand.\n"
          "• Eingaben nur in den gelben Zellen im Blatt „Bewertung“."),
]


def write_codebook(ws):
    ws.column_dimensions["A"].width = 16
    ws.column_dimensions["B"].width = 34
    ws.column_dimensions["C"].width = 90
    r = 1
    for kind, content in CODEBOOK:
        if kind == "title":
            ws.cell(r, 1, content).font = F_TITLE
            r += 2
        elif kind == "h":
            c = ws.cell(r, 1, content)
            c.font = F_BOLD
            c.fill = FILL_LABEL
            ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=3)
            r += 1
        elif kind == "p":
            c = ws.cell(r, 1, content)
            c.font, c.alignment = F_BODY, WRAP
            ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=3)
            ws.row_dimensions[r].height = row_height(content) * 0.95
            r += 2
        elif kind == "scale":
            for head, col in (("Wert", 1), ("Bedeutung", 2), ("Anteil belegter Aussagen", 3)):
                c = ws.cell(r, col, head)
                c.font, c.border = F_BOLD, BOX
            r += 1
            for val, name, desc in content:
                for col, txt in ((1, val), (2, name), (3, desc)):
                    c = ws.cell(r, col, txt)
                    c.font, c.border, c.alignment = F_BODY, BOX, WRAP
                r += 1
            r += 1
        elif kind == "example":
            for label, txt in content.items():
                a = ws.cell(r, 1, label)
                a.font, a.fill, a.alignment = F_BOLD, FILL_EXAMPLE, WRAP
                b = ws.cell(r, 2, txt)
                b.font, b.fill, b.alignment = F_BODY, FILL_EXAMPLE, WRAP
                ws.merge_cells(start_row=r, start_column=2, end_row=r, end_column=3)
                ws.row_dimensions[r].height = min(409, 13.5 * max(1, -(-len(txt) // 115)) + 4)
                r += 1
            r += 1


def write_rating_sheet(ws, items: pd.DataFrame) -> dict:
    ws.column_dimensions["A"].width = 22
    ws.column_dimensions["B"].width = 120
    ws.sheet_view.zoomScale = 110
    dv_rating = DataValidation(type="list", formula1='"1,2,3,4,5"', allow_blank=True,
                               showErrorMessage=True, errorTitle="Ungültiger Wert",
                               error="Bitte einen Wert von 1 bis 5 wählen.")
    dv_unsure = DataValidation(type="list", formula1='"ja,nein"', allow_blank=True)
    ws.add_data_validation(dv_rating)
    ws.add_data_validation(dv_unsure)

    legend = ws.cell(1, 1, "Eingaben nur in den gelben Zellen: „Bewertung (1–5)“ per Dropdown, "
                           "„Unsicher?“ und optional „Kommentar“. Anleitung und Skala: Blatt „Codebook“.")
    legend.font, legend.alignment = F_BOLD, WRAP
    ws.merge_cells("A1:B1")
    ws.row_dimensions[1].height = 30

    cells = {}
    r = 3
    n = len(items)
    for _, it in items.iterrows():
        head = ws.cell(r, 1, f"Eintrag {it['item_id']}  ({int(it['item_id'][1:])} / {n})")
        head.font, head.fill = F_HEAD, FILL_HEAD
        ws.cell(r, 2).fill = FILL_HEAD
        r += 1

        blocks = [("Frage", str(it["question"]))]
        for i, ctx in enumerate(it["_ctx"], 1):
            blocks.append((f"Textstelle {i}", str(ctx).strip()))
        blocks.append(("Antwort des Systems", clean_answer(it["answer"])))
        for label, txt in blocks:
            a = ws.cell(r, 1, label)
            a.font, a.fill, a.alignment, a.border = F_BOLD, FILL_LABEL, WRAP, BOX
            b = ws.cell(r, 2, txt)
            b.font, b.alignment, b.border = (F_BOLD if label == "Antwort des Systems" else F_BODY), WRAP, BOX
            ws.row_dimensions[r].height = row_height(txt)
            r += 1

        for label, dv in (("Bewertung (1–5)", dv_rating), ("Unsicher?", dv_unsure), ("Kommentar", None)):
            a = ws.cell(r, 1, label)
            a.font, a.fill, a.border = F_BOLD, FILL_LABEL, BOX
            b = ws.cell(r, 2)
            b.fill, b.border, b.font, b.alignment = FILL_INPUT, BOX, F_BODY, WRAP
            if dv is not None:
                dv.add(b)
            if label == "Kommentar":
                ws.row_dimensions[r].height = 30
            cells.setdefault(it["item_id"], {})[label] = b.coordinate
            r += 1
        r += 1
    ws.freeze_panes = "A2"
    return cells


def build_workbook(items: pd.DataFrame, path: str) -> dict:
    wb = Workbook()
    cb = wb.active
    cb.title = "Codebook"
    write_codebook(cb)
    ws = wb.create_sheet("Bewertung")
    cells = write_rating_sheet(ws, items)
    wb.active = 1
    wb.save(path)
    return cells


def main():
    pool = load_pool()
    items = sample(pool)
    cells = None
    for rater in ("A", "B"):
        cells = build_workbook(items, os.path.join(HERE, f"Rater_{rater}.xlsx"))

    key = items[["item_id", "source", "pipeline", "file", "row", "stratum", "faithfulness",
                 "question"]].rename(columns={"faithfulness": "ragas_faithfulness"})
    key["rating_cell"] = key["item_id"].map(lambda i: cells[i]["Bewertung (1–5)"])
    key["unsure_cell"] = key["item_id"].map(lambda i: cells[i]["Unsicher?"])
    key["comment_cell"] = key["item_id"].map(lambda i: cells[i]["Kommentar"])
    key.to_csv(os.path.join(HERE, "key_DO_NOT_SHARE.csv"), index=False)

    print(f"{len(items)} Eintraege, Pool {len(pool)}")
    print(key.groupby(["source", "stratum"]).size().to_string())
    print(key[key.source == "LLM"]["pipeline"].value_counts().to_string())
    print("RAGAS F: mean={:.3f}, min={:.2f}, max={:.2f}".format(
        key.ragas_faithfulness.mean(), key.ragas_faithfulness.min(), key.ragas_faithfulness.max()))


if __name__ == "__main__":
    main()
