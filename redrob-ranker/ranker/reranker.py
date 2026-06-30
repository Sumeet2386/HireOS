"""
Cross-encoder reranking for top-K precision optimization.

Inserts a high-precision reranking stage between honeypot pruning and
final output.  Uses a lightweight cross-encoder that concatenates
``[JD_TEXT, CANDIDATE_TEXT]`` and passes the joint sequence through
full transformer self-attention to produce a relevance score.

Unlike dual-encoders (FAISS/BGE) which embed query and document
independently, cross-encoders capture deep inter-token dependencies
between the job description and the candidate profile.  This is the
single most impactful technique for NDCG@10 improvement.

Default model: ``mixedbread-ai/mxbai-rerank-xsmall-v1`` (70M params),
which runs in ~400ms-3s for 100 pairs on CPU.

Precompute mode: scores can be pre-computed offline and stored in
``artifacts/cross_encoder_scores.json`` to avoid loading the model
during the inference phase.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Precomputed score path (avoids loading the model during inference)
# ---------------------------------------------------------------------------
_DEFAULT_PRECOMPUTED_PATH = "artifacts/cross_encoder_scores.json"


def load_precomputed_scores(
    path: str | Path = _DEFAULT_PRECOMPUTED_PATH,
) -> dict[str, float] | None:
    """Load pre-computed cross-encoder scores from disk.

    Parameters
    ----------
    path : str or Path
        Path to the JSON file mapping candidate_id → score.

    Returns
    -------
    dict[str, float] or None
        Score mapping, or ``None`` if the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        logger.debug("No precomputed cross-encoder scores at %s", path)
        return None

    with open(path, "r", encoding="utf-8") as fh:
        scores = json.load(fh)

    logger.info("Loaded %d precomputed cross-encoder scores", len(scores))
    return scores


def _build_candidate_text(candidate: dict[str, Any]) -> str:
    """Build a rich text representation of a candidate for cross-encoding.

    Includes profile summary, skills, and career history -- the same
    information a human recruiter would scan when evaluating fit.

    Parameters
    ----------
    candidate : dict
        Full candidate record from ``candidates.jsonl``.

    Returns
    -------
    str
        Concatenated text representation (truncated to fit model context).
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])

    parts = [
        f"Title: {profile.get('current_title', 'Unknown')}",
        f"Experience: {profile.get('years_of_experience', 0):.1f} years",
        f"Company: {profile.get('current_company', 'Unknown')} ({profile.get('current_industry', '')})",
        f"Location: {profile.get('location', '')}, {profile.get('country', '')}",
    ]

    # Headline and summary (most information-dense)
    headline = profile.get("headline", "")
    if headline:
        parts.append(f"Headline: {headline}")

    summary = profile.get("summary", "")
    if summary:
        parts.append(f"Summary: {summary[:300]}")

    # Top skills with proficiency
    skill_strs = []
    for s in skills[:15]:
        name = s.get("name", "")
        prof = s.get("proficiency", "")
        if name:
            skill_strs.append(f"{name} ({prof})" if prof else name)
    if skill_strs:
        parts.append(f"Skills: {', '.join(skill_strs)}")

    # Career history (most recent 3 roles)
    for job in career[:3]:
        title = job.get("title", "")
        company = job.get("company", "")
        duration = job.get("duration_months", 0)
        desc = (job.get("description") or "")[:250]
        parts.append(f"Role: {title} at {company} ({duration}mo): {desc}")

    return " | ".join(parts)


def rerank_with_cross_encoder(
    jd_text: str,
    candidates: list[tuple[str, dict[str, Any]]],
    model_name: str = "mixedbread-ai/mxbai-rerank-xsmall-v1",
    batch_size: int = 32,
) -> list[tuple[str, float]]:
    """Rerank candidates using a cross-encoder model.

    Loads the model, builds ``(jd_text, candidate_text)`` pairs, and
    scores them via full transformer cross-attention.

    Parameters
    ----------
    jd_text : str
        Full job description text.
    candidates : list[tuple[str, dict]]
        List of ``(candidate_id, candidate_record)`` pairs.
    model_name : str
        HuggingFace model identifier for the cross-encoder.
    batch_size : int
        Batch size for inference.

    Returns
    -------
    list[tuple[str, float]]
        List of ``(candidate_id, cross_encoder_score)`` sorted
        descending by score.
    """
    from sentence_transformers import CrossEncoder

    logger.info(
        "Loading cross-encoder '%s' for %d candidates...",
        model_name, len(candidates),
    )
    model = CrossEncoder(model_name, max_length=512)

    pairs = []
    cids = []
    for cid, cand in candidates:
        candidate_text = _build_candidate_text(cand)
        pairs.append((jd_text, candidate_text))
        cids.append(cid)

    logger.info("Scoring %d pairs (batch_size=%d)...", len(pairs), batch_size)
    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)

    results = list(zip(cids, scores.tolist()))
    results.sort(key=lambda x: (-x[1], x[0]))

    logger.info(
        "Cross-encoder reranking complete. Score range: %.4f to %.4f",
        min(s for _, s in results),
        max(s for _, s in results),
    )

    return results


def rerank_candidates(
    jd_text: str,
    candidate_pool: list[tuple[str, float]],
    candidates_by_id: dict[str, dict[str, Any]],
    precomputed_scores: dict[str, float] | None = None,
    top_k: int = 100,
    rerank_depth: int = 150,
    cross_encoder_weight: float = 0.6,
    original_score_weight: float = 0.4,
) -> list[tuple[str, float]]:
    """Rerank candidates using cross-encoder scores (precomputed or live).

    Blends cross-encoder scores with the original pipeline scores for
    robustness.  If precomputed scores are available, uses those
    (zero latency).  Otherwise, falls back to live cross-encoder inference.

    Parameters
    ----------
    jd_text : str
        Full job description text.
    candidate_pool : list[tuple[str, float]]
        ``(candidate_id, original_score)`` from the hand-tuned formula,
        already pruned of honeypots.
    candidates_by_id : dict[str, dict]
        Full candidate records by ID.
    precomputed_scores : dict[str, float] or None
        Optional mapping of ``candidate_id → cross_encoder_score``.
    top_k : int
        Number of final candidates to return.
    rerank_depth : int
        How many candidates from ``candidate_pool`` to rerank.
    cross_encoder_weight : float
        Weight for cross-encoder score in the blend (default 0.6).
    original_score_weight : float
        Weight for original pipeline score in the blend (default 0.4).

    Returns
    -------
    list[tuple[str, float]]
        Reranked ``(candidate_id, blended_score)`` list, length ``top_k``.
    """
    # Take top rerank_depth candidates
    pool = candidate_pool[:rerank_depth]

    if precomputed_scores:
        # Use precomputed scores (zero latency)
        logger.info("Using precomputed cross-encoder scores for %d candidates", len(pool))

        # Normalize original scores to [0, 1]
        orig_scores = [s for _, s in pool]
        orig_min = min(orig_scores) if orig_scores else 0
        orig_max = max(orig_scores) if orig_scores else 1
        orig_range = max(orig_max - orig_min, 1e-9)

        results = []
        for cid, orig_score in pool:
            ce_score = precomputed_scores.get(cid, 0.0)
            orig_norm = (orig_score - orig_min) / orig_range
            blended = cross_encoder_weight * ce_score + original_score_weight * orig_norm
            results.append((cid, blended))

    else:
        # Live cross-encoder inference
        live_candidates = []
        for cid, _ in pool:
            cand = candidates_by_id.get(cid)
            if cand:
                live_candidates.append((cid, cand))

        if not live_candidates:
            logger.warning("No candidates available for cross-encoder reranking")
            return pool[:top_k]

        ce_results = rerank_with_cross_encoder(jd_text, live_candidates)
        ce_scores = {cid: score for cid, score in ce_results}

        # Normalize both score sets to [0, 1] for blending
        orig_scores = [s for _, s in pool]
        orig_min = min(orig_scores) if orig_scores else 0
        orig_max = max(orig_scores) if orig_scores else 1
        orig_range = max(orig_max - orig_min, 1e-9)

        ce_vals = list(ce_scores.values())
        ce_min = min(ce_vals) if ce_vals else 0
        ce_max = max(ce_vals) if ce_vals else 1
        ce_range = max(ce_max - ce_min, 1e-9)

        results = []
        for cid, orig_score in pool:
            orig_norm = (orig_score - orig_min) / orig_range
            ce_raw = ce_scores.get(cid, 0.0)
            ce_norm = (ce_raw - ce_min) / ce_range
            blended = cross_encoder_weight * ce_norm + original_score_weight * orig_norm
            results.append((cid, blended))

    # Sort by blended score descending
    results.sort(key=lambda x: (-x[1], x[0]))

    logger.info(
        "Reranking complete: %d → %d candidates (CE weight=%.1f, orig weight=%.1f)",
        len(pool), min(top_k, len(results)),
        cross_encoder_weight, original_score_weight,
    )

    return results[:top_k]
