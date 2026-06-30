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

    Used to produce normalized feature values for downstream scoring
    (``cosine_similarity_jd``, ``bm25_score_jd``).  No longer used for
    fusion -- see :func:`hybrid_recall` which now uses RRF.

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


# Reciprocal Rank Fusion smoothing constant.
# k=60 is the empirically established default across IR benchmarks
# (Cormack et al., 2009).  Higher values flatten rank differences;
# lower values amplify top-rank contributions.
RRF_K: int = 60


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

    Uses **Reciprocal Rank Fusion (RRF)** instead of min-max
    normalization for score combination.  RRF is rank-based rather
    than score-based, providing:

    - Complete distribution agnosticism (dense and sparse score
      distributions need not be comparable).
    - Outlier immunity (a keyword-stuffed candidate with an extreme
      BM25 score cannot compress other candidates' scores).
    - Consensus reward (candidates appearing in the top ranks of
      *both* retrieval lists are naturally promoted).

    The original min-max normalized scores are still preserved in the
    returned tuples for use as downstream feature columns
    (``cosine_similarity_jd``, ``bm25_score_jd``).

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
        Legacy parameter (kept for API compatibility). Not used by RRF.
    sparse_weight : float
        Legacy parameter (kept for API compatibility). Not used by RRF.

    Returns
    -------
    list[tuple[str, float, float]]
        List of ``(candidate_id, dense_score_norm, sparse_score_norm)``
        sorted by RRF score descending.  Union of both retrieval sets.
    """
    # -- Dense retrieval --
    dense_scores, dense_indices = dense_recall(jd_embedding, faiss_index, k_dense)
    dense_norm = _normalize_scores(dense_scores)

    # -- Sparse retrieval --
    sparse_scores, sparse_indices = sparse_recall(jd_text, bm25_index, k_sparse)
    sparse_norm = _normalize_scores(sparse_scores)

    # Build normalized-score dicts for downstream feature use
    dense_norm_dict: dict[int, float] = {}
    for idx, score in zip(dense_indices.tolist(), dense_norm.tolist()):
        if idx >= 0:  # FAISS returns -1 for invalid
            dense_norm_dict[idx] = score

    sparse_norm_dict: dict[int, float] = {}
    for idx, score in zip(sparse_indices.tolist(), sparse_norm.tolist()):
        sparse_norm_dict[idx] = score

    # Build rank dicts for RRF (1-indexed: position 0 → rank 1)
    dense_rank: dict[int, int] = {}
    for rank_pos, idx in enumerate(dense_indices.tolist()):
        if idx >= 0:
            dense_rank[idx] = rank_pos + 1

    sparse_rank: dict[int, int] = {}
    for rank_pos, idx in enumerate(sparse_indices.tolist()):
        sparse_rank[idx] = rank_pos + 1

    # -- RRF fusion --
    # RRF(d) = Σ_r  1 / (k + rank_r(d))
    # Candidates only in one list get a single RRF term.
    all_indices = set(dense_rank.keys()) | set(sparse_rank.keys())

    results = []
    for idx in all_indices:
        rrf_score = 0.0
        if idx in dense_rank:
            rrf_score += 1.0 / (RRF_K + dense_rank[idx])
        if idx in sparse_rank:
            rrf_score += 1.0 / (RRF_K + sparse_rank[idx])

        cand_id = id_mapping.get(idx, f"UNKNOWN_{idx}")
        d_norm = dense_norm_dict.get(idx, 0.0)
        s_norm = sparse_norm_dict.get(idx, 0.0)
        results.append((cand_id, d_norm, s_norm, rrf_score))

    # Sort by RRF score descending, break ties by candidate_id
    results.sort(key=lambda x: (-x[3], x[0]))

    logger.info(
        "Hybrid recall (RRF k=%d): %d dense + %d sparse = %d unique candidates",
        RRF_K,
        len(dense_rank),
        len(sparse_rank),
        len(all_indices),
    )

    return [(cid, d, s) for cid, d, s, _ in results]
