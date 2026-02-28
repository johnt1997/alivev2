# make_bm25_jsonl.py
import os
import json
import argparse
from packages.document_processing import DocumentProcessing
from packages.globals import EMBEDDINGS

# Konfiguration über Command-Line
parser = argparse.ArgumentParser()
parser.add_argument("--person", default="Washington", help="Name der Person (Ordner unter ./data/)")
parser.add_argument("--splitter", required=True, choices=["recursive", "sentence_transformer", "semantic"], help="Splitter-Typ")
parser.add_argument("--chunk_size", type=int, required=True, help="Chunk Größe (Zeichen/Tokens)")
parser.add_argument("--chunk_overlap", type=int, required=True, help="Chunk Overlap (Zeichen/Tokens)")
parser.add_argument("--include_size_in_path", action="store_true", help="Include chunk_size/overlap in output path")
args = parser.parse_args()

# Input/Output Pfade
INPUT_DIR = f"./data/{args.person}"
if args.include_size_in_path:
    OUTPUT_DIR = f"./bm25_jsonl/{args.person}_{args.splitter}_{args.chunk_size}_{args.chunk_overlap}"
else:
    OUTPUT_DIR = f"./bm25_jsonl/{args.person}_{args.splitter}"  # WICHTIG: _splitter suffix!
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dokumente chunken
print(f"[INFO] Chunking with {args.splitter} (size={args.chunk_size}, overlap={args.chunk_overlap})")
dp = DocumentProcessing(embeddings=EMBEDDINGS)
chunks = dp.get_chunked_documents(
    directory_path=INPUT_DIR,
    splitter_type=args.splitter,
    chunk_size=args.chunk_size,
    chunk_overlap=args.chunk_overlap
)

# JSONL schreiben
output_path = os.path.join(OUTPUT_DIR, "docs.jsonl")  # Muss genau 'docs.jsonl' heißen!
with open(output_path, "w", encoding="utf-8") as f:
    for chunk in chunks:
        record = {
            "id": chunk.metadata["source"],  # Muss mit FAISS metadata['source'] übereinstimmen
            "contents": chunk.page_content
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"[SUCCESS] Wrote {len(chunks)} chunks to {output_path}")