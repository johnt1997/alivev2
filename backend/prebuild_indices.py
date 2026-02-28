import subprocess, sys, os

# === KONFIGURATION ===
PERSON = "Washington"  # Hier ändern für neuen Korpus, z.B. "eu_geschichte"

CONFIGS = [
    {"splitter": "semantic", "chunk_size": 0, "chunk_overlap": 0},
    {"splitter": "sentence_transformer", "chunk_size": 256, "chunk_overlap": 30},
    {"splitter": "recursive", "chunk_size": 1000, "chunk_overlap": 30}
]

os.environ["TOKENIZERS_PARALLELISM"] = "false"

for cfg in CONFIGS:
    # 1) JSONL
    rc = subprocess.run(
        f"python make_bm25_jsonl.py --person {PERSON} "
        f"--splitter {cfg['splitter']} --chunk_size {cfg['chunk_size']} --chunk_overlap {cfg['chunk_overlap']}",
        shell=True
    ).returncode
    if rc != 0:
        sys.exit(f"[FAIL] JSONL build failed for {cfg}")

    # 2) INDEX BAUEN
    cmd = (
        "python -m pyserini.index.lucene "
        "--collection JsonCollection "
        f"--input ./bm25_jsonl/{PERSON}_{cfg['splitter']} "
        f"--index ./bm25_indexes/{PERSON}_{cfg['splitter']} "
        "--generator DefaultLuceneDocumentGenerator "
        "--threads 4 "
        "--storePositions --storeDocvectors --storeRaw"
    )
    rc = subprocess.run(cmd, shell=True).returncode
    if rc != 0:
        sys.exit(f"[FAIL] Index build failed for {cfg}")

    # 3) VALIDIEREN
    from pyserini.search.lucene import LuceneSearcher
    idx_dir = f"./bm25_indexes/{PERSON}_{cfg['splitter']}"
    s = LuceneSearcher(idx_dir)
    print(f"[OK] Index ready: {idx_dir} | num_docs={s.num_docs}")

