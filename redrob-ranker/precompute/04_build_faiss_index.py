"""
04_build_faiss_index.py — Build FAISS IndexFlatIP from embeddings.

Exact inner-product search (cosine similarity on unit-normed vectors).

Usage:
    python precompute/04_build_faiss_index.py --embeddings <npy_path> --out <output_dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import faiss
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS index.")
    parser.add_argument("--embeddings", required=True, help="Path to embeddings.npy")
    parser.add_argument("--out", default="artifacts", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load embeddings
    print(f"Loading embeddings from {args.embeddings}...")
    embeddings = np.load(args.embeddings)
    n, dim = embeddings.shape
    print(f"Loaded {n:,} vectors of dimension {dim}")

    # Ensure float32 and contiguous
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

    # Build IndexFlatIP (exact inner product)
    print("Building FAISS IndexFlatIP...")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"Index contains {index.ntotal:,} vectors")

    # Save
    output_path = output_dir / "index.faiss"
    faiss.write_index(index, str(output_path))
    size_mb = output_path.stat().st_size / 1e6
    print(f"Saved FAISS index to {output_path} ({size_mb:.1f} MB)")

    # Quick sanity check
    print("\nSanity check: querying index with first vector...")
    query = embeddings[0:1]
    scores, indices = index.search(query, 5)
    print(f"Top 5 results for vector[0]: indices={indices[0].tolist()}, scores={scores[0].tolist()}")
    assert indices[0][0] == 0, "First result should be the query vector itself!"
    print("OK: Sanity check passed.")


if __name__ == "__main__":
    main()
