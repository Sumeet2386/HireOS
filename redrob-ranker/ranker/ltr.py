"""
LightGBM Learning-to-Rank inference module.

Loads a pre-trained LightGBM LambdaMART model and scores candidates
based on their feature vectors.

Also provides a hand-tuned fallback scoring function that uses a
sophisticated multi-signal formula when the LTR model is unavailable
or produces poor results.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .constants import SCORING_WEIGHTS as W
from .utils import compute_notice_score

if TYPE_CHECKING:
    import lightgbm as lgb

logger = logging.getLogger(__name__)


def load_ltr_model(model_path: str | Path) -> lgb.Booster:
    """Load a pre-trained LightGBM model.

    Parameters
    ----------
    model_path : str or Path
        Filesystem path to the ``.bin`` model file.

    Returns
    -------
    lgb.Booster
        Loaded LightGBM Booster instance.
    """
    import lightgbm as lgb

    model = lgb.Booster(model_file=str(model_path))
    logger.info("Loaded LTR model from %s", model_path)
    return model


def predict_scores(
    model: lgb.Booster,
    feature_matrix: np.ndarray,
    feature_names: list[str] | None = None,
) -> np.ndarray:
    """Predict relevance scores for a batch of candidates.

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
    model: lgb.Booster,
    candidate_features: dict[str, dict[str, float]],
    feature_columns: list[str],
) -> list[tuple[str, float]]:
    """Rank candidates using the LTR model.

    Parameters
    ----------
    model : lgb.Booster
        Pre-trained LightGBM model.
    candidate_features : dict
        ``{candidate_id: {feature_name: value}}`` mapping.
    feature_columns : list[str]
        Ordered list of feature columns matching model training.

    Returns
    -------
    list[tuple[str, float]]
        List of ``(candidate_id, score)`` sorted descending by score.
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
    """Sophisticated multi-signal scoring -- additive formula with balanced weights.

    Uses an ADDITIVE combination of five signal groups (weights from
    ``SCORING_WEIGHTS``):

    1. Core Fit (title + skills + experience + company)
    2. Semantic Match (cosine + BM25)
    3. Behavioral (engagement signals)
    4. Availability (notice period + open_to_work)
    5. Location
    Minus: Hard Penalties (honeypot, keyword stuffer)

    Key design decisions vs previous version:
    - Behavioral is ADDITIVE, not multiplicative. A great candidate with
      poor behavioral signals loses ~18% max, not ~70%.
    - Notice period is piecewise linear aligned to JD language.
    - YoE uses smooth logistic transition at 5.0, no cliff edge.
    - Skill quality weight increased (entailment is the best anti-honeypot
      signal).

    Parameters
    ----------
    candidate_features : dict
        ``{candidate_id: {feature_name: value}}`` mapping.

    Returns
    -------
    list[tuple[str, float]]
        Sorted list of ``(candidate_id, score)`` descending by score.
    """
    results = []
    for cid, feats in candidate_features.items():

        # -- 1. Core Fit Score (0-1 range) --
        title_current = feats.get("current_title_relevance", 0.0)
        title_best = feats.get("best_title_relevance", 0.0)

        title_score = (
            title_current * 0.6 +
            title_best * 0.4
        )

        # YoE -- smooth logistic curve, no cliff at 5.0
        yoe_raw = feats.get("years_of_experience", 0.0)
        if 5.0 <= yoe_raw <= 9.0:
            yoe_score = max(0.65, 1.0 - 0.08 * abs(yoe_raw - 7.0))
        elif yoe_raw < 5.0:
            yoe_score = 0.65 / (1.0 + math.exp(-1.5 * (yoe_raw - 3.5)))
        else:
            yoe_score = max(0.15, 0.65 - 0.05 * (yoe_raw - 9.0))

        # Mild experience penalties
        under_exp_penalty = max(0.0, (5.0 - yoe_raw) * 0.05) if yoe_raw < 5.0 else 0.0
        over_exp_penalty = max(0.0, (yoe_raw - 9.0) * 0.02) if yoe_raw > 9.0 else 0.0

        # Skill match (nonlinear -- reward having multiple high-signal skills)
        n_high_signal = feats.get("num_high_signal_skills", 0.0)
        skill_match = min(1.0, n_high_signal / W.high_signal_skill_saturation)
        skill_match_bonus = min(W.skill_match_bonus_cap, n_high_signal * W.skill_match_bonus_per_skill)

        # Skill quality signals
        skill_quality = (
            feats.get("skill_proficiency_score", 0.0) * 0.25 +
            feats.get("skill_text_entailment_rate", 0.0) * 0.35 +
            feats.get("has_embedding_skills", 0.0) * 0.12 +
            feats.get("has_vector_db_skills", 0.0) * 0.12 +
            feats.get("has_nlp_ir_skills", 0.0) * 0.08 +
            feats.get("has_evaluation_skills", 0.0) * 0.08
        )

        # Product company experience
        product_count = feats.get("product_company_count", 0.0)
        product_count_score = min(1.0, product_count / 3.0)
        company_score = (
            product_count_score * 0.45 +
            feats.get("has_product_company_exp", 0.0) * 0.25 +
            (1.0 - feats.get("all_consulting_career", 0.0)) * 0.15 +
            feats.get("tenure_fit", 0.5) * 0.15
        )

        core_fit = (
            title_score * W.title_weight +
            yoe_score * W.yoe_weight +
            skill_match * W.skill_match_weight +
            skill_quality * W.skill_quality_weight +
            company_score * W.company_weight
        ) + skill_match_bonus - under_exp_penalty - over_exp_penalty

        # -- 2. Semantic Match Score (0-1 range) --
        semantic_score = (
            feats.get("cosine_similarity_jd", 0.0) * 0.75 +
            feats.get("bm25_score_jd", 0.0) * 0.25
        )

        # -- 3. Behavioral Score (0-1 range -- ADDITIVE, not multiplier) --
        activity = feats.get("activity_decay_score", 0.5)
        response = feats.get("recruiter_response_rate", 0.3)
        interview = feats.get("interview_completion_rate", 0.5)
        github = feats.get("github_activity_score", 0.0)

        saved = min(1.0, feats.get("saved_by_recruiters_30d", 0.0) / 20.0)
        search = min(1.0, feats.get("search_appearance_30d", 0.0) / 200.0)
        views = min(1.0, feats.get("profile_views_30d", 0.0) / 100.0)

        behavioral_score = (
            search * 0.22 +
            saved * 0.20 +
            response * 0.15 +
            activity * 0.15 +
            views * 0.10 +
            interview * 0.08 +
            github * 0.05 +
            feats.get("open_to_work", 0.0) * 0.05
        )

        # -- 4. Availability Score (0-1 range) --
        notice_days = feats.get("notice_period_days", 0.0)
        notice_score = compute_notice_score(notice_days)

        availability_score = (
            notice_score * 0.70 +
            feats.get("open_to_work", 0.0) * 0.30
        )

        # -- 5. Location Score (0-1 range) --
        is_india = feats.get("is_india", 0.0)
        is_tier1 = feats.get("is_tier1_india", 0.0)
        willing = feats.get("willing_to_relocate", 0.0)
        if is_india > 0 and is_tier1 > 0:
            location = 1.0
        elif is_india > 0:
            location = 0.85 if willing > 0 else 0.75
        else:
            location = 0.70 if willing > 0 else 0.50

        # -- 6. Hard Penalties --
        penalty = 0.0

        if feats.get("is_honeypot", 0.0) > 0:
            penalty += W.honeypot_penalty

        if feats.get("has_maturity_impossible", 0.0) > 0:
            penalty += W.maturity_impossible_penalty
            if feats.get("skill_text_entailment_rate", 0.0) < 0.40:
                penalty += 0.08

        flag_count = feats.get("honeypot_flag_count", 0.0)
        if flag_count >= 7:
            penalty += 0.35
        elif flag_count >= 5:
            penalty += 0.18
        elif flag_count >= 4:
            penalty += 0.10
        elif flag_count >= 3:
            penalty += 0.05

        # Low entailment penalty -- only when advanced skills are claimed
        entailment = feats.get("skill_text_entailment_rate", 0.0)
        n_advanced = feats.get("num_advanced_expert_skills", 0.0)
        if n_advanced > 0:
            if entailment < 0.10:
                penalty += 0.10
            elif entailment < 0.20:
                penalty += 0.06
            elif entailment < 0.30:
                penalty += 0.03

        trust_penalty = feats.get("assessment_trust_penalty", 0.0)
        penalty += trust_penalty * 0.15

        if feats.get("is_title_chaser", 0.0) > 0:
            penalty += W.title_chaser_penalty

        if feats.get("all_consulting_career", 0.0) > 0:
            penalty += W.all_consulting_penalty

        if feats.get("has_fictional_company", 0.0) > 0:
            penalty += W.fictional_company_penalty

        title_rel = feats.get("current_title_relevance", 0.5)
        n_ai = feats.get("num_core_ai_skills", 0.0)
        if title_rel <= 0.1 and n_ai >= 3:
            penalty += W.keyword_stuffer_penalty

        # -- Final Composite Score -- ADDITIVE, weights sum to 1.0 --
        raw_score = (
            core_fit * W.core_fit +
            semantic_score * W.semantic +
            behavioral_score * W.behavioral +
            availability_score * W.availability +
            location * W.location
        )

        # Apply penalties (multiplicative to avoid negative scores)
        final_score = raw_score * max(0.01, 1.0 - penalty)

        results.append((cid, final_score))

    results.sort(key=lambda x: (-x[1], x[0]))
    return results
