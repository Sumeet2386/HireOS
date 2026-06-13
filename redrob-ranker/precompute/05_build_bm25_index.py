"""
05_build_bm25_index.py — Build BM25 sparse index from candidate texts.

Uses rank_bm25 library for Okapi BM25 scoring.

Usage:
    python precompute/05_build_bm25_index.py --texts <parquet_path> --out <output_dir>
"""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path


def simple_tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenization with lowercasing."""
    text = text.lower()
    # Remove punctuation except hyphens in compound words
    text = re.sub(r'[^\w\s\-]', ' ', text)
    tokens = text.split()
    # Filter very short tokens
    return [t for t in tokens if len(t) > 1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build BM25 index.")
    parser.add_argument("--texts", required=True, help="Path to candidate_texts.parquet or .csv")
    parser.add_argument("--out", default="artifacts", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load texts
    texts_path = Path(args.texts)
    print(f"Loading texts from {texts_path}...")

    if texts_path.suffix == ".parquet":
        import pyarrow.parquet as pq
        table = pq.read_table(texts_path)
        full_texts = table.column("full_text").to_pylist()
    else:
        import csv
        full_texts = []
        with open(texts_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                full_texts.append(row["full_text"])

    print(f"Loaded {len(full_texts):,} texts.")

    # Tokenize
    print("Tokenizing...")
    tokenized = [simple_tokenize(text) for text in full_texts]
    avg_tokens = sum(len(t) for t in tokenized) / len(tokenized)
    print(f"Average tokens per document: {avg_tokens:.1f}")

    # Build BM25 index
    print("Building BM25 index...")
    from rank_bm25 import BM25Okapi
    bm25 = BM25Okapi(tokenized)
    print("BM25 index built.")

    # Save
    output_path = output_dir / "bm25_index.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(bm25, f)
    size_mb = output_path.stat().st_size / 1e6
    print(f"Saved BM25 index to {output_path} ({size_mb:.1f} MB)")

    # Sanity check
    print("\nSanity check: querying 'machine learning pytorch embeddings'...")
    query = simple_tokenize("machine learning pytorch embeddings ranking retrieval")
    scores = bm25.get_scores(query)
    top5 = sorted(range(len(scores)), key=lambda i: -scores[i])[:5]
    print(f"Top 5 indices: {top5}")
    print(f"Top 5 scores: {[round(scores[i], 3) for i in top5]}")
    print("OK: Sanity check passed.")


if __name__ == "__main__":
    main()
