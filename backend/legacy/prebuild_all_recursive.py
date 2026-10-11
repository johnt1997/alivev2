"""
Baut ALLE BM25-Indices für recursive Splitter neu.
Notwendig nach dem Fix von sliding-window → standard RecursiveCharacterTextSplitter.

Baut 4 Indices:
  1. Washington_recursive         (1000/30) - für Chapter 5 Kompatibilität (1_evaluate_rag_base.ipynb)
  2. Washington_recursive_500_30  (500/30)  - Chunk-Size Experiment
  3. Washington_recursive_1000_30 (1000/30) - Chunk-Size Experiment
  4. Washington_recursive_1500_50 (1500/50) - Chunk-Size Experiment

Usage:
    cd backend
    python prebuild_all_recursive.py
"""
import subprocess
import sys
import os
import shutil

PERSON = "Washington"
PYTHON = sys.executable  # use same Python that launched this script

# ALL recursive configs that need BM25 indices
CONFIGS = [
    # Chapter 5 base path (without chunk_size in path)
    {"splitter": "recursive", "chunk_size": 1000, "chunk_overlap": 30, "include_size": False},
    # Chunk-size experiment paths (with chunk_size in path)
    {"splitter": "recursive", "chunk_size": 500,  "chunk_overlap": 30, "include_size": True},
    {"splitter": "recursive", "chunk_size": 1000, "chunk_overlap": 30, "include_size": True},
    {"splitter": "recursive", "chunk_size": 1500, "chunk_overlap": 50, "include_size": True},
]

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["JAVA_HOME"] = r"C:\Program Files\Eclipse Adoptium\jdk-17.0.18.8-hotspot"

# Step 0: Delete old recursive FAISS databases (will be auto-rebuilt by notebooks)
print("=" * 60)
print("[STEP 0] Cleaning old recursive FAISS databases")
print("=" * 60)
db_base = f"./database/{PERSON}"
if os.path.isdir(db_base):
    for d in os.listdir(db_base):
        if d.startswith("recursive_"):
            path = os.path.join(db_base, d)
            print(f"  Deleting: {path}")
            shutil.rmtree(path)
    print("[OK] Old recursive FAISS databases removed")
else:
    print(f"[INFO] No database dir found at {db_base}")

for i, cfg in enumerate(CONFIGS):
    s = cfg["splitter"]
    cs = cfg["chunk_size"]
    co = cfg["chunk_overlap"]
    include_size = cfg["include_size"]

    if include_size:
        suffix = f"{PERSON}_{s}_{cs}_{co}"
    else:
        suffix = f"{PERSON}_{s}"

    print(f"\n{'='*60}")
    print(f"[{i+1}/{len(CONFIGS)}] Building BM25 for: {suffix} (size={cs}, overlap={co})")
    print(f"{'='*60}")

    # Delete old index if exists
    old_jsonl = f"./bm25_jsonl/{suffix}"
    old_index = f"./bm25_indexes/{suffix}"
    if os.path.isdir(old_index):
        print(f"  Removing old index: {old_index}")
        shutil.rmtree(old_index)
    if os.path.isdir(old_jsonl):
        print(f"  Removing old JSONL: {old_jsonl}")
        shutil.rmtree(old_jsonl)

    # Step 1: Build JSONL
    cmd_jsonl = (
        f"{PYTHON} make_bm25_jsonl.py --person {PERSON} "
        f"--splitter {s} --chunk_size {cs} --chunk_overlap {co}"
    )
    if include_size:
        cmd_jsonl += " --include_size_in_path"

    print(f"  [1/2] Building JSONL...")
    rc = subprocess.run(cmd_jsonl, shell=True).returncode
    if rc != 0:
        sys.exit(f"[FAIL] JSONL build failed for {suffix}")

    # Step 2: Build Lucene index
    jsonl_dir = f"./bm25_jsonl/{suffix}"
    index_dir = f"./bm25_indexes/{suffix}"

    cmd_index = (
        f"{PYTHON} -m pyserini.index.lucene "
        "--collection JsonCollection "
        f"--input {jsonl_dir} "
        f"--index {index_dir} "
        "--generator DefaultLuceneDocumentGenerator "
        "--threads 4 "
        "--storePositions --storeDocvectors --storeRaw"
    )
    print(f"  [2/2] Building Lucene index...")
    rc = subprocess.run(cmd_index, shell=True).returncode
    if rc != 0:
        sys.exit(f"[FAIL] Index build failed for {suffix}")

    # Validate
    try:
        from pyserini.search.lucene import LuceneSearcher
        searcher = LuceneSearcher(index_dir)
        print(f"  [OK] Index ready: {index_dir} | num_docs={searcher.num_docs}")
    except Exception as e:
        print(f"  [WARN] Could not validate: {e}")

print(f"\n{'='*60}")
print("[DONE] All recursive BM25 indices rebuilt with fixed splitter.")
print("Next: Run 3_rerun_recursive.ipynb")
print(f"{'='*60}")
