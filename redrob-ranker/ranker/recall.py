"""
Hybrid candidate recall: FAISS dense + BM25 sparse retrieval.

Stage 1 of the ranking pipeline: 100K -> ~5,000 unique candidates.

Dense retrieval (FAISS):
    - Pre-built IndexFlatIP with bge-small-en-v1.5 embeddings (384d)
    - Top-K inner product search for semantic similarity

Sparse retrieval (BM25):
    - Pre-built BM25 index on tokenized candidate texts
    - Top-K BM25 scoring for lexical matching

Results are unioned and scores normalized to [0,1] for combination.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import faiss as faiss_module
    from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


def load_faiss_index(index_path: str | Path) -> faiss_module.Index:
    """Load a pre-built FAISS index.

    Parameters
    ----------
    index_path : str or Path
        Filesystem path to the ``.faiss`` index file.

    Returns
    -------
    faiss.Index
        Loaded FAISS index.

    .. warning::
        FAISS indices are loaded via C++ deserialization. Ensure the
        index file originates from a trusted source.
    """
    import faiss

    index = faiss.read_index(str(index_path))
    logger.info("Loaded FAISS index with %d vectors", index.ntotal)
    return index


def load_bm25_index(index_path: str | Path) -> BM25Okapi:
    """Load a pre-built BM25 index from pickle.

    Parameters
    ----------
    index_path : str or Path
        Filesystem path to the pickled BM25 index.

    Returns
    -------
    BM25Okapi
        Loaded BM25 index.

    .. warning::
        Uses ``pickle.load`` which can execute arbitrary code. Only load
        pickle files from trusted sources. Consider verifying file
        checksums before loading in production.
    """
    with open(index_path, "rb") as fh:
        index = pickle.load(fh)  # noqa: S301
    logger.info("Loaded BM25 index from %s", index_path)
    return index


def load_id_mapping(mapping_path: str | Path) -> dict[int, str]:
    """Load row-index to candidate_id mapping.

    Parameters
    ----------
    mapping_path : str or Path
        Path to the JSON mapping file.

    Returns
    -------
    dict[int, str]
        Mapping of integer row indices to candidate ID strings.
    """
    with open(mapping_path, "r", encoding="utf-8") as fh:
        mapping = json.load(fh)
    # Convert string keys back to int
    result = {int(k): v for k, v in mapping.items()}
    logger.info("Loaded ID mapping with %d entries", len(result))
    return result


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Normalize scores to [0, 1] using min-max scaling.

    Parameters
    ----------
    scores : np.ndarray
        Raw score array.

    Returns
    -------
    np.ndarray
        Normalized scores (zeros if range is negligible).
    """
    min_s = scores.min()
    max_s = scores.max()
    if max_s - min_s < 1e-9:
        return np.zeros_like(scores)
    return (scores - min_s) / (max_s - min_s)


def dense_recall(
    jd_embedding: np.ndarray,
    faiss_index: faiss_module.Index,
    k: int = 5000,
) -> tuple[np.ndarray, np.ndarray]:
    """Dense retrieval using FAISS inner product search.

    Parameters
    ----------
    jd_embedding : np.ndarray
        JD query embedding, shape ``(1, dim)`` or ``(dim,)``.
    faiss_index : faiss.Index
        Pre-built FAISS IndexFlatIP.
    k : int
        Number of top candidates to retrieve.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(scores, indices)`` -- both shape ``(k,)``.
    """
    if jd_embedding.ndim == 1:
        jd_embedding = jd_embedding.reshape(1, -1)

    jd_embedding = jd_embedding.astype(np.float32)
    scores, indices = faiss_index.search(jd_embedding, k)
    return scores[0], indices[0]


def sparse_recall(
    jd_text: str,
    bm25_index: BM25Okapi,
    k: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """Sparse retrieval using BM25 scoring.

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
        ``(scores, indices)`` -- both shape ``(k,)``.
    """
    query_tokens = jd_text.lower().split()
    all_scores = bm25_index.get_scores(query_tokens)

    top_indices = np.argsort(all_scores)[-k:][::-1]
    top_scores = all_scores[top_indices]

    return top_scores, top_indices


def hybrid_recall(
    jd_embedding: np.ndarray,
    jd_text: str,
    faiss_index: faiss_module.Index,
    bm25_index: BM25Okapi,
    id_mapping: dict[int, str],
    k_dense: int = 5000,
    k_sparse: int = 500,
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
) -> list[tuple[str, float, float]]:
    """Hybrid recall combining FAISS dense + BM25 sparse retrieval.

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
        Row-index to candidate_id.
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
        List of ``(candidate_id, dense_score, sparse_score)`` sorted by
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

    logger.info(
        "Hybrid recall: %d dense + %d sparse = %d unique candidates",
        len(dense_dict),
        len(sparse_dict),
        len(all_indices),
    )

    return [(cid, d, s) for cid, d, s, _ in results]
