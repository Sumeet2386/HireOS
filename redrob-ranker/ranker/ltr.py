"""
LightGBM Learning-to-Rank inference module.

Loads a pre-trained LightGBM LambdaMART model and scores candidates
based on their feature vectors.
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
    Fallback scoring using hand-tuned weights if LTR model is unavailable.

    This implements a weighted linear combination of key features,
    calibrated to approximate the LTR model's behavior.
    """
    WEIGHTS = {
        # Semantic (added online)
        "cosine_similarity_jd": 0.20,
        "bm25_score_jd": 0.05,
        # Structural
        "yoe_in_ideal_range": 0.08,
        "current_title_relevance": 0.10,
        "best_title_relevance": 0.05,
        "has_product_company_exp": 0.04,
        "all_consulting_career": -0.06,
        "is_title_chaser": -0.03,
        "tenure_fit": 0.02,
        # Skills
        "num_high_signal_skills": 0.08,
        "skill_proficiency_score": 0.04,
        "skill_text_entailment_rate": 0.03,
        "has_embedding_skills": 0.03,
        "has_vector_db_skills": 0.02,
        "has_nlp_ir_skills": 0.03,
        "has_evaluation_skills": 0.02,
        "assessment_trust_penalty": -0.04,
        # Behavioral
        "activity_decay_score": 0.04,
        "recruiter_response_rate": 0.03,
        "notice_period_multiplier": 0.03,
        "interview_completion_rate": 0.02,
        "github_activity_score": 0.01,
        # Location
        "location_fit": 0.05,
        # Honeypot
        "is_honeypot": -0.50,  # Heavy penalty
    }

    results = []
    for cid, feats in candidate_features.items():
        score = 0.0
        for feat_name, weight in WEIGHTS.items():
            value = feats.get(feat_name, 0.0)
            score += weight * value
        results.append((cid, score))

    results.sort(key=lambda x: (-x[1], x[0]))
    return results
