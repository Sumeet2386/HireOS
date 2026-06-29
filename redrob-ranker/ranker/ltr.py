"""
LightGBM Learning-to-Rank inference module.

Loads a pre-trained LightGBM LambdaMART model and scores candidates
based on their feature vectors.

Also provides a hand-tuned fallback scoring function that uses a
sophisticated multi-signal formula when the LTR model is unavailable
or produces poor results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def load_ltr_model(model_path: str | Path):
    """Load a pre-trained LightGBM model."""
    import lightgbm as lgb
    model = lgb.Booster(model_file=str(model_path))
    return model


def predict_scores(
    model: Any,
    feature_matrix: np.ndarray,
    feature_names: list[str] | None = None,
) -> np.ndarray:
    """
    Predict relevance scores for a batch of candidates.

    Parameters
    ----------
    model : lgb.Booster
        Pre-trained LightGBM model.
    feature_matrix : np.ndarray
        Feature matrix of shape (n_candidates, n_features).
    feature_names : list[str], optional
        Feature names matching the model's training features.

    Returns
    -------
    np.ndarray
        Predicted relevance scores, shape (n_candidates,).
    """
    scores = model.predict(feature_matrix)
    return scores


def rank_candidates(
    model: Any,
    candidate_features: dict[str, dict[str, float]],
    feature_columns: list[str],
) -> list[tuple[str, float]]:
    """
    Rank candidates using the LTR model.

    Parameters
    ----------
    model : lgb.Booster
        Pre-trained LightGBM model.
    candidate_features : dict
        {candidate_id: {feature_name: value}} mapping.
    feature_columns : list[str]
        Ordered list of feature columns matching model training.

    Returns
    -------
    list[tuple[str, float]]
        List of (candidate_id, score) sorted descending by score.
    """
    cand_ids = list(candidate_features.keys())
    n = len(cand_ids)

    # Build feature matrix
    feature_matrix = np.zeros((n, len(feature_columns)), dtype=np.float32)
    for i, cid in enumerate(cand_ids):
        feats = candidate_features[cid]
        for j, col in enumerate(feature_columns):
            feature_matrix[i, j] = feats.get(col, 0.0)

    scores = predict_scores(model, feature_matrix)

    # Pair with candidate IDs and sort
    results = list(zip(cand_ids, scores.tolist()))
    results.sort(key=lambda x: (-x[1], x[0]))  # Score desc, ID asc for ties

    return results


def fallback_weighted_scoring(
    candidate_features: dict[str, dict[str, float]],
) -> list[tuple[str, float]]:
    """
    Sophisticated multi-signal scoring — additive formula with balanced weights.

    Uses an ADDITIVE combination of five signal groups (weights sum to 1.0):
      1. Core Fit (title + skills + experience + company) — 42%
      2. Semantic Match (cosine + BM25)                  — 25%
      3. Behavioral (engagement signals)                 — 18%
      4. Availability (notice period + open_to_work)     —  7%
      5. Location                                        —  8%
      Subtotal: 100%
      Minus: Hard Penalties (honeypot, keyword stuffer)  — up to -100%

    Key design decisions vs previous version:
      - Behavioral is ADDITIVE, not multiplicative. A great candidate with
        poor behavioral signals loses ~18% max, not ~70%.
      - Notice period is piecewise linear aligned to JD language:
        ≤30d = no penalty, 30-60d = gentle, 60-120d = moderate.
      - YoE uses smooth logistic transition at 5.0, no cliff edge.
      - Skill quality weight increased (entailment is the best anti-honeypot signal).
    """
    import math

    results = []
    for cid, feats in candidate_features.items():

        # ── 1. Core Fit Score (0-1 range, 48% weight) ──
        title_current = feats.get("current_title_relevance", 0.0)
        title_best = feats.get("best_title_relevance", 0.0)

        # best_title_relevance already captures career history strength.
        # No override needed — let the 40% weight on best_title do its job.
        title_score = (
            title_current * 0.6 +
            title_best * 0.4
        )

        # YoE — smooth logistic curve, no cliff at 5.0
        yoe_raw = feats.get("years_of_experience", 0.0)
        if 5.0 <= yoe_raw <= 9.0:
            # In ideal range: score peaks at 7, gentle decay to edges
            yoe_score = max(0.65, 1.0 - 0.08 * abs(yoe_raw - 7.0))
        elif yoe_raw < 5.0:
            # Below ideal: smooth logistic transition (no cliff!)
            # 3.0→0.30, 4.0→0.43, 4.5→0.50, 4.9→0.56, 5.0→0.65
            yoe_score = 0.65 / (1.0 + math.exp(-1.5 * (yoe_raw - 3.5)))
        else:
            # Above ideal (>9): gentle decay
            yoe_score = max(0.15, 0.65 - 0.05 * (yoe_raw - 9.0))

        # Mild under-experience penalty (much softer than before)
        under_exp_penalty = max(0.0, (5.0 - yoe_raw) * 0.05) if yoe_raw < 5.0 else 0.0

        # Mild over-experience penalty
        over_exp_penalty = max(0.0, (yoe_raw - 9.0) * 0.02) if yoe_raw > 9.0 else 0.0

        # Skill match (nonlinear — reward having multiple high-signal skills)
        n_high_signal = feats.get("num_high_signal_skills", 0.0)
        skill_match = min(1.0, n_high_signal / 6.0)  # Saturates at 6 skills
        skill_match_bonus = min(0.25, n_high_signal * 0.04)  # Extra bonus per skill

        # Skill quality signals — INCREASED WEIGHT (was 0.12, now 0.18)
        skill_quality = (
            feats.get("skill_proficiency_score", 0.0) * 0.25 +
            feats.get("skill_text_entailment_rate", 0.0) * 0.35 +  # Best anti-honeypot
            feats.get("has_embedding_skills", 0.0) * 0.12 +  # JD core requirement
            feats.get("has_vector_db_skills", 0.0) * 0.12 +  # JD core requirement
            feats.get("has_nlp_ir_skills", 0.0) * 0.08 +
            feats.get("has_evaluation_skills", 0.0) * 0.08
        )

        # Product company experience (JD explicitly prefers this)
        company_score = (
            feats.get("has_product_company_exp", 0.0) * 0.6 +
            (1.0 - feats.get("all_consulting_career", 0.0)) * 0.2 +
            feats.get("tenure_fit", 0.5) * 0.2
        )

        core_fit = (
            title_score * 0.22 +       # Reduced — surface signal
            yoe_score * 0.20 +          # Reduced — surface signal
            skill_match * 0.15 +
            skill_quality * 0.25 +      # Increased — best quality signal
            company_score * 0.18        # Increased — JD prefers product co.
        ) + skill_match_bonus - under_exp_penalty - over_exp_penalty

        # ── 2. Semantic Match Score (0-1 range, 25% weight) ──
        semantic_score = (
            feats.get("cosine_similarity_jd", 0.0) * 0.75 +
            feats.get("bm25_score_jd", 0.0) * 0.25
        )

        # ── 3. Behavioral Score (0-1 range, 18% weight — ADDITIVE, not multiplier) ──
        activity = feats.get("activity_decay_score", 0.5)
        response = feats.get("recruiter_response_rate", 0.3)
        interview = feats.get("interview_completion_rate", 0.5)
        github = feats.get("github_activity_score", 0.0)

        behavioral_score = (
            activity * 0.30 +
            response * 0.30 +
            interview * 0.15 +
            github * 0.10 +
            feats.get("open_to_work", 0.0) * 0.15
        )

        # ── 4. Availability Score (0-1 range, 7% weight) ──
        # Notice period — piecewise linear aligned with JD language
        notice_days = feats.get("notice_period_days", 0.0)
        if notice_days <= 30:
            notice_score = 1.0                                       # JD-preferred
        elif notice_days <= 45:
            notice_score = 1.0 - 0.005 * (notice_days - 30)         # 45d → 0.925
        elif notice_days <= 60:
            notice_score = 0.925 - 0.005 * (notice_days - 45)       # 60d → 0.85
        elif notice_days <= 90:
            notice_score = 0.85 - 0.005 * (notice_days - 60)        # 90d → 0.70
        elif notice_days <= 120:
            notice_score = 0.70 - 0.005 * (notice_days - 90)        # 120d → 0.55
        else:
            notice_score = max(0.3, 0.55 - 0.003 * (notice_days - 120))

        availability_score = (
            notice_score * 0.70 +
            feats.get("open_to_work", 0.0) * 0.30
        )

        # ── 5. Location Score (0-1 range, 8% weight) ──
        # IMPORTANT: Recompute from raw features because precomputed
        # features.parquet has OLD location values (0.32 for intl unwilling).
        # The pipeline uses precomputed features (rank.py line 178-179),
        # so the features.py fix is NOT applied unless we override here.
        is_india = feats.get("is_india", 0.0)
        is_tier1 = feats.get("is_tier1_india", 0.0)
        willing = feats.get("willing_to_relocate", 0.0)
        if is_india > 0 and is_tier1 > 0:
            location = 1.0
        elif is_india > 0:
            location = 0.85 if willing > 0 else 0.75
        else:
            location = 0.70 if willing > 0 else 0.50

        # ── 6. Hard Penalties ──
        penalty = 0.0

        # Honeypot: near-zero score
        if feats.get("is_honeypot", 0.0) > 0:
            penalty += 0.95
            
        # Maturity Impossible penalty (strong signal of profile inflation)
        if feats.get("has_maturity_impossible", 0.0) > 0:
            penalty += 0.20
            # Compound: maturity impossible + low entailment = very suspicious
            if feats.get("skill_text_entailment_rate", 0.0) < 0.40:
                penalty += 0.08

        # Honeypot flag count — smoother gradient (was only 4/6 tiers)
        flag_count = feats.get("honeypot_flag_count", 0.0)
        if flag_count >= 7:
            penalty += 0.35
        elif flag_count >= 5:
            penalty += 0.18
        elif flag_count >= 4:
            penalty += 0.10
        elif flag_count >= 3:
            penalty += 0.05

        # LOW ENTAILMENT PENALTY — the critical anti-honeypot signal.
        # Only apply when the candidate actually HAS advanced/expert skills.
        # Candidates with all-intermediate skills get entailment_rate=0.0
        # naturally, and shouldn't be penalized for it.
        entailment = feats.get("skill_text_entailment_rate", 0.0)
        n_advanced = feats.get("num_advanced_expert_skills", 0.0)
        if n_advanced > 0:  # Only penalize if they claim advanced skills
            if entailment < 0.10:
                penalty += 0.10  # Severe: virtually no skill backing
            elif entailment < 0.20:
                penalty += 0.06  # Very low: most skills unsubstantiated
            elif entailment < 0.30:
                penalty += 0.03  # Low: concerning but not disqualifying

        # Assessment trust penalty (claims advanced but scores poorly)
        trust_penalty = feats.get("assessment_trust_penalty", 0.0)
        penalty += trust_penalty * 0.15

        # Title chaser penalty
        if feats.get("is_title_chaser", 0.0) > 0:
            penalty += 0.05

        # All-consulting career penalty
        if feats.get("all_consulting_career", 0.0) > 0:
            penalty += 0.08

        # ── Final Composite Score — ADDITIVE, weights sum to 1.0 ──
        raw_score = (
            core_fit * 0.48 +           # Increased: skills+title matter most
            semantic_score * 0.25 +
            behavioral_score * 0.12 +   # Reduced: platform engagement ≠ quality
            availability_score * 0.07 +
            location * 0.08
        )

        # Apply penalties (multiplicative to avoid negative scores)
        final_score = raw_score * max(0.01, 1.0 - penalty)

        results.append((cid, final_score))

    results.sort(key=lambda x: (-x[1], x[0]))
    return results
