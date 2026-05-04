#!/usr/bin/env python3
"""
Generiert Evaluations-Fragen für einen Corpus mit OpenAI.

Beispiel:
    python create_questions.py --corpus Eu --output hilfreich2.csv --n 42
"""

import os
import argparse
import json
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_openai import ChatOpenAI
import pandas as pd


def generate_questions_from_chunks(chunks: list, llm, n_questions: int) -> list:
    """Generiert Fragen aus Text-Chunks mit GPT."""

    # Kombiniere alle Chunks zu einem Kontext-String (max 50k chars)
    full_text = "\n\n---\n\n".join([c.page_content[:2000] for c in chunks[:50]])

    prompt = f"""Based on the following document content about European Union history, generate exactly {n_questions} diverse evaluation questions with ground truth answers.

Requirements:
- Mix of factual, reasoning, and multi-hop questions
- Each question should be answerable from the provided content
- Provide accurate ground truth answers based on the content
- Questions should be in English

Content:
{full_text[:40000]}

Return as JSON array with objects containing "question", "ground_truth", and "question_type" (one of: "factual", "reasoning", "multi_hop").

JSON:"""

    response = llm.invoke(prompt)

    # Parse JSON from response
    try:
        # Find JSON array in response
        text = response.content
        start = text.find('[')
        end = text.rfind(']') + 1
        if start >= 0 and end > start:
            questions = json.loads(text[start:end])
            return questions
    except json.JSONDecodeError:
        print("[WARN] Could not parse JSON, retrying...")

    return []


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation questions")
    parser.add_argument("--corpus", required=True, help="Corpus name (folder in data/)")
    parser.add_argument("--output", default="hilfreich.csv", help="Output filename")
    parser.add_argument("--n", type=int, default=42, help="Number of questions to generate")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model for generation")
    args = parser.parse_args()

    # 1. PDFs laden
    data_path = f"./data/{args.corpus}"
    print(f"[INFO] Loading PDFs from {data_path}...")
    loader = PyPDFDirectoryLoader(data_path)
    docs = loader.load()
    print(f"[INFO] Loaded {len(docs)} pages")

    # 2. Fragen generieren
    print(f"[INFO] Generating {args.n} questions with {args.model}...")
    llm = ChatOpenAI(model=args.model, temperature=0.7)

    questions = generate_questions_from_chunks(docs, llm, args.n)

    if len(questions) < args.n:
        print(f"[WARN] Only got {len(questions)} questions, generating more...")
        more = generate_questions_from_chunks(docs[len(docs)//2:], llm, args.n - len(questions))
        questions.extend(more)

    print(f"[INFO] Generated {len(questions)} questions")

    # 3. DataFrame erstellen
    df = pd.DataFrame(questions)

    # Ensure required columns exist
    if 'question' not in df.columns:
        print("[ERROR] No questions generated!")
        return

    if 'ground_truth' not in df.columns:
        df['ground_truth'] = ""
    if 'question_type' not in df.columns:
        df['question_type'] = "factual"

    # 4. Speichern
    output_dir = f"./autogen_questions/{args.corpus}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{args.output}"
    df.to_csv(output_path, index=True)
    print(f"[SUCCESS] Saved {len(df)} questions to {output_path}")

    # Preview
    print("\n[PREVIEW] First 3 questions:")
    for i, row in df.head(3).iterrows():
        print(f"  Q{i+1}: {row['question'][:80]}...")


if __name__ == "__main__":
    main()
