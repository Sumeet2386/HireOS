"""
03_generate_embeddings.py — Generate bge-small-en-v1.5 embeddings for all candidates.

Uses sentence-transformers with GPU acceleration for batch encoding.
Normalizes vectors to unit length for cosine similarity via inner product.

Usage:
    python precompute/03_generate_embeddings.py --texts <parquet_path> --out <output_dir> [--batch-size 256]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate embeddings.")
    parser.add_argument("--texts", required=True, help="Path to candidate_texts.parquet or .csv")
    parser.add_argument("--out", default="artifacts", help="Output directory")
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5",
                        help="Sentence-transformers model name")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for encoding")
    parser.add_argument("--device", default=None, help="Device: 'cuda', 'cpu', or None for auto")
    args = parser.parse_args()

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load texts
    texts_path = Path(args.texts)
    print(f"Loading texts from {texts_path}...")

    if texts_path.suffix == ".parquet":
        import pyarrow.parquet as pq
        table = pq.read_table(texts_path)
        texts = table.column("full_text").to_pylist()
    else:
        import csv
        texts = []
        with open(texts_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                texts.append(row["full_text"])

    print(f"Loaded {len(texts):,} texts.")

    # Load model
    print(f"Loading model: {args.model}...")
    from sentence_transformers import SentenceTransformer

    device = args.device
    if device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SentenceTransformer(args.model, device=device)
    print(f"Model loaded on {device}. Embedding dimension: {model.get_sentence_embedding_dimension()}")

    # Generate embeddings
    print(f"Encoding {len(texts):,} texts with batch_size={args.batch_size}...")
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # Unit-normalize for cosine via inner product
        convert_to_numpy=True,
    )

    embeddings = embeddings.astype(np.float32)
    print(f"Embeddings shape: {embeddings.shape}")

    # Save
    output_path = output_dir / "embeddings.npy"
    np.save(output_path, embeddings)
    print(f"Saved embeddings to {output_path} ({embeddings.nbytes / 1e6:.1f} MB)")

    # Also generate JD embedding
    print("\nGenerating JD embedding...")
    jd_text = (
        "Senior AI Engineer for a founding team building an AI-powered talent intelligence "
        "platform. Must have production experience with embeddings, semantic search, retrieval "
        "systems, ranking and recommendation systems, vector databases (FAISS, Pinecone, Weaviate), "
        "NLP, transformers, and evaluation frameworks (NDCG, MRR, MAP). "
        "Looking for a shipper over a researcher — someone who builds, deploys, and iterates in "
        "production. Ideal candidate has 5-9 years of experience, works with Python, PyTorch or "
        "TensorFlow, and has shipped ML/AI systems in product companies. "
        "India-based preferred (Pune, Noida, Hyderabad, Mumbai, Delhi NCR). "
        "Red flags: pure consulting career, title chaser, no recent coding, superficial AI keywords."
    )

    jd_embedding = model.encode(
        [jd_text],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    jd_path = output_dir / "jd_embedding.npy"
    np.save(jd_path, jd_embedding)
    print(f"Saved JD embedding to {jd_path}")


if __name__ == "__main__":
    main()
