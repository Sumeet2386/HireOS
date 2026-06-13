"""
Hybrid candidate recall: FAISS dense + BM25 sparse retrieval.

Stage 1 of the ranking pipeline: 100K → ~5,000 unique candidates.

Dense retrieval (FAISS):
    - Pre-built IndexFlatIP with bge-small-en-v1.5 embeddings (384d)
    - Top-K inner product search for semantic similarity

Sparse retrieval (BM25):
    - Pre-built BM25 index on tokenized candidate texts
    - Top-K BM25 scoring for lexical matching

Results are unioned and scores normalized to [0,1] for combination.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np


def load_faiss_index(index_path: str | Path):
    """Load a pre-built FAISS index."""
    import faiss
    index = faiss.read_index(str(index_path))
    return index


def load_bm25_index(index_path: str | Path):
    """Load a pre-built BM25 index from pickle."""
    with open(index_path, "rb") as f:
        return pickle.load(f)


def load_id_mapping(mapping_path: str | Path) -> dict[int, str]:
    """Load row-index to candidate_id mapping."""
    import json
    with open(mapping_path, "r") as f:
        mapping = json.load(f)
    # Convert string keys back to int
    return {int(k): v for k, v in mapping.items()}


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Normalize scores to [0, 1] using min-max scaling."""
    min_s = scores.min()
    max_s = scores.max()
    if max_s - min_s < 1e-9:
        return np.zeros_like(scores)
    return (scores - min_s) / (max_s - min_s)


def dense_recall(
    jd_embedding: np.ndarray,
    faiss_index: Any,
    k: int = 5000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Dense retrieval using FAISS inner product search.

    Parameters
    ----------
    jd_embedding : np.ndarray
        JD query embedding, shape (1, dim) or (dim,).
    faiss_index : faiss.Index
        Pre-built FAISS IndexFlatIP.
    k : int
        Number of top candidates to retrieve.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (scores, indices) — both shape (k,)
    """
    if jd_embedding.ndim == 1:
        jd_embedding = jd_embedding.reshape(1, -1)

    # Ensure float32
    jd_embedding = jd_embedding.astype(np.float32)

    scores, indices = faiss_index.search(jd_embedding, k)
    return scores[0], indices[0]


def sparse_recall(
    jd_text: str,
    bm25_index: Any,
    k: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sparse retrieval using BM25 scoring.

    Parameters
    ----------
    jd_text : str
        Job description text for query.
    bm25_index : BM25Okapi
        Pre-built BM25 index.
    k : int
        Number of top candidates to retrieve.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (scores, indices) — both shape (k,)
    """
    # Simple whitespace tokenization
    query_tokens = jd_text.lower().split()
    all_scores = bm25_index.get_scores(query_tokens)

    # Get top-k indices
    top_indices = np.argsort(all_scores)[-k:][::-1]
    top_scores = all_scores[top_indices]

    return top_scores, top_indices


def hybrid_recall(
    jd_embedding: np.ndarray,
    jd_text: str,
    faiss_index: Any,
    bm25_index: Any,
    id_mapping: dict[int, str],
    k_dense: int = 5000,
    k_sparse: int = 500,
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
) -> list[tuple[str, float, float]]:
    """
    Hybrid recall combining FAISS dense + BM25 sparse retrieval.

    Parameters
    ----------
    jd_embedding : np.ndarray
        JD query embedding.
    jd_text : str
        JD text for BM25.
    faiss_index : faiss.Index
        Pre-built FAISS index.
    bm25_index : BM25Okapi
        Pre-built BM25 index.
    id_mapping : dict
        Row-index → candidate_id.
    k_dense : int
        Number of FAISS results.
    k_sparse : int
        Number of BM25 results.
    dense_weight : float
        Weight for dense scores in final combination.
    sparse_weight : float
        Weight for sparse scores in final combination.

    Returns
    -------
    list[tuple[str, float, float]]
        List of (candidate_id, dense_score, sparse_score) sorted by
        combined weighted score descending. Union of both retrieval sets.
    """
    # Dense retrieval
    dense_scores, dense_indices = dense_recall(jd_embedding, faiss_index, k_dense)
    dense_norm = _normalize_scores(dense_scores)

    # Sparse retrieval
    sparse_scores, sparse_indices = sparse_recall(jd_text, bm25_index, k_sparse)
    sparse_norm = _normalize_scores(sparse_scores)

    # Build score dictionaries keyed by row index
    dense_dict: dict[int, float] = {}
    for idx, score in zip(dense_indices.tolist(), dense_norm.tolist()):
        if idx >= 0:  # FAISS returns -1 for invalid
            dense_dict[idx] = score

    sparse_dict: dict[int, float] = {}
    for idx, score in zip(sparse_indices.tolist(), sparse_norm.tolist()):
        sparse_dict[idx] = score

    # Union of both sets
    all_indices = set(dense_dict.keys()) | set(sparse_dict.keys())

    results = []
    for idx in all_indices:
        d_score = dense_dict.get(idx, 0.0)
        s_score = sparse_dict.get(idx, 0.0)
        combined = dense_weight * d_score + sparse_weight * s_score
        cand_id = id_mapping.get(idx, f"UNKNOWN_{idx}")
        results.append((cand_id, d_score, s_score, combined))

    # Sort by combined score descending
    results.sort(key=lambda x: -x[3])

    return [(cid, d, s) for cid, d, s, _ in results]
