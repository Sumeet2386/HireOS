"""
rank.py -- Main inference script for the Redrob AI Candidate Ranking System.

Must run in <=5 min, <=16GB RAM, CPU only, no network.

Orchestrates the multi-stage pipeline:

1. Load pre-computed artifacts (FAISS index, BM25 index, features, etc.)
2. Hybrid recall (FAISS + BM25): 100K -> ~5,000 unique candidates
3. Feature engineering + multi-signal scoring: ~5,000 -> 300
4. Honeypot pruning: 300 -> ~100 clean candidates
5. Reasoning generation: top 100 with per-candidate justifications
6. CSV output + validation

Usage::

    python rank.py --candidates ./candidates.jsonl --out ./submission.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

# Ensure ranker package is importable
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ranker.constants import JD_BM25_KEYWORDS, PIPELINE_CONFIG as CFG
from ranker.features import extract_all_features
from ranker.honeypot import detect_honeypot
from ranker.ltr import fallback_weighted_scoring
from ranker.reasoning import generate_reasoning
from ranker.utils import extract_honeypot_flag_features, load_candidates
from ranker.validator import validate_submission

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_artifacts(artifacts_dir: Path) -> dict:
    """Load all pre-computed artifacts from disk.

    Parameters
    ----------
    artifacts_dir : Path
        Directory containing pre-computed artifact files.

    Returns
    -------
    dict
        Dictionary of loaded artifacts keyed by name.
    """
    artifacts = {}
    t0 = time.time()

    # FAISS index
    faiss_path = artifacts_dir / "index.faiss"
    if faiss_path.exists():
        import faiss
        artifacts["faiss_index"] = faiss.read_index(str(faiss_path))
        logger.info("FAISS index: %s vectors", f"{artifacts['faiss_index'].ntotal:,}")
    else:
        logger.warning("FAISS index not found at %s. Skipping dense recall.", faiss_path)

    # BM25 index
    bm25_path = artifacts_dir / "bm25_index.pkl"
    if bm25_path.exists():
        with open(bm25_path, "rb") as fh:
            artifacts["bm25_index"] = pickle.load(fh)  # noqa: S301
        logger.info("BM25 index loaded.")
    else:
        logger.warning("BM25 index not found at %s. Skipping sparse recall.", bm25_path)

    # JD embedding
    jd_path = artifacts_dir / "jd_embedding.npy"
    if jd_path.exists():
        emb = np.load(jd_path).astype(np.float32)
        if emb.ndim < 1:
            raise ValueError(f"JD embedding has unexpected shape: {emb.shape}")
        artifacts["jd_embedding"] = emb
        logger.info("JD embedding: shape %s", emb.shape)

    # ID mapping
    mapping_path = artifacts_dir / "id_mapping.json"
    if mapping_path.exists():
        with open(mapping_path, "r", encoding="utf-8") as fh:
            artifacts["id_mapping"] = {int(k): v for k, v in json.load(fh).items()}
        logger.info("ID mapping: %s entries", f"{len(artifacts['id_mapping']):,}")

    # Features (pre-computed)
    features_path = artifacts_dir / "features.parquet"
    if features_path.exists():
        import pyarrow.parquet as pq
        table = pq.read_table(features_path)
        df_dict = table.to_pydict()
        cids = df_dict.pop("candidate_id")
        feature_cols = list(df_dict.keys())
        artifacts["precomputed_features"] = {
            cid: {col: df_dict[col][i] for col in feature_cols}
            for i, cid in enumerate(cids)
        }
        logger.info(
            "Pre-computed features: %s candidates, %d features",
            f"{len(artifacts['precomputed_features']):,}",
            len(feature_cols),
        )

    elapsed = time.time() - t0
    logger.info("Artifacts loaded in %.1fs", elapsed)

    return artifacts


def _enrich_honeypot_features(
    cid: str,
    feats: dict[str, float],
    candidate: dict,
) -> int:
    """Run live honeypot detection and merge flags into feature dict.

    Returns 1 if the candidate was newly flagged, 0 otherwise.
    """
    is_hp, flags = detect_honeypot(candidate)
    new_hp = 1.0 if is_hp else 0.0
    flag_features = extract_honeypot_flag_features(flags)

    enriched = 0
    if new_hp > feats.get("is_honeypot", 0.0):
        feats["is_honeypot"] = new_hp
        enriched = 1

    # Never reduce flag counts -- take the maximum of precomputed vs live
    for key, value in flag_features.items():
        feats[key] = max(feats.get(key, 0.0), value)

    return enriched


def run_pipeline(
    candidates_path: str,
    output_path: str,
    artifacts_dir: str = "artifacts",
) -> None:
    """Run the full ranking pipeline.

    Parameters
    ----------
    candidates_path : str
        Path to the candidate JSONL file.
    output_path : str
        Output CSV path.
    artifacts_dir : str
        Directory containing pre-computed artifacts.
    """
    total_start = time.time()
    artifacts_dir_path = Path(artifacts_dir)

    logger.info("=" * 60)
    logger.info("REDROB AI CANDIDATE RANKING SYSTEM")
    logger.info("=" * 60)

    # -- Stage 0: Load artifacts --
    logger.info("[Stage 0] Loading artifacts...")
    artifacts = load_artifacts(artifacts_dir_path)

    # -- Stage 1: Load candidates --
    logger.info("[Stage 1] Loading candidates from %s...", candidates_path)
    t1 = time.time()
    candidates_by_id = load_candidates(candidates_path)
    n_total = len(candidates_by_id)
    logger.info("Loaded %s candidates in %.1fs", f"{n_total:,}", time.time() - t1)

    # -- Stage 2: Recall + feature extraction --
    logger.info("[Stage 2] Candidate recall + feature extraction...")
    t2 = time.time()

    has_recall = "faiss_index" in artifacts and "jd_embedding" in artifacts
    precomputed = artifacts.get("precomputed_features", {})

    if has_recall:
        from ranker.recall import hybrid_recall
        recall_results = hybrid_recall(
            jd_embedding=artifacts["jd_embedding"],
            jd_text=JD_BM25_KEYWORDS,
            faiss_index=artifacts["faiss_index"],
            bm25_index=artifacts.get("bm25_index"),
            id_mapping=artifacts["id_mapping"],
            k_dense=CFG.k_dense,
            k_sparse=CFG.k_sparse if "bm25_index" in artifacts else 0,
        )
        recalled_ids = [cid for cid, _, _ in recall_results[:CFG.k_dense]]
        dense_scores = {cid: ds for cid, ds, _ in recall_results}
        sparse_scores = {cid: ss for cid, _, ss in recall_results}
        logger.info("Recalled %s candidates via hybrid retrieval", f"{len(recalled_ids):,}")
    else:
        recalled_ids = list(candidates_by_id.keys())
        dense_scores = {}
        sparse_scores = {}
        logger.info("No recall artifacts. Scoring all %s candidates.", f"{len(recalled_ids):,}")

    # Build feature vectors for recalled candidates
    candidate_features: dict[str, dict[str, float]] = {}
    for cid in recalled_ids:
        if cid in precomputed:
            features = dict(precomputed[cid])
        elif cid in candidates_by_id:
            features = extract_all_features(candidates_by_id[cid])
        else:
            continue

        # Add semantic features (from recall)
        features["cosine_similarity_jd"] = dense_scores.get(cid, 0.0)
        features["bm25_score_jd"] = sparse_scores.get(cid, 0.0)

        candidate_features[cid] = features

    # Enrich precomputed features with live honeypot checks
    enriched = 0
    for cid, feats in candidate_features.items():
        if cid in candidates_by_id:
            enriched += _enrich_honeypot_features(cid, feats, candidates_by_id[cid])

    logger.info(
        "Features extracted for %s candidates in %.1fs",
        f"{len(candidate_features):,}",
        time.time() - t2,
    )
    if enriched > 0:
        logger.info("Live honeypot enrichment: %d candidates newly flagged", enriched)

    # -- Stage 3: Multi-signal Scoring --
    logger.info("[Stage 3] Multi-signal scoring...")
    t3 = time.time()

    logger.info("Using multi-signal weighted scoring formula.")
    ranked = fallback_weighted_scoring(candidate_features)
    logger.info("Scored %s candidates in %.1fs", f"{len(ranked):,}", time.time() - t3)

    # -- Stage 4: Honeypot pruning --
    logger.info("[Stage 4] Honeypot pruning (top %d -> top %d)...", CFG.top_n_for_pruning, CFG.final_output_size)
    t4 = time.time()

    top_candidates = ranked[:CFG.top_n_for_pruning]
    clean_candidates: list[tuple[str, float]] = []
    pruned_count = 0

    for cid, score in top_candidates:
        feats = candidate_features.get(cid, {})

        # Run live honeypot check if candidate data is available
        if cid in candidates_by_id:
            _enrich_honeypot_features(cid, feats, candidates_by_id[cid])

        if feats.get("is_honeypot", 0) > 0:
            pruned_count += 1
            continue

        clean_candidates.append((cid, score))

    logger.info("Pruned %d honeypots from top %d", pruned_count, CFG.top_n_for_pruning)

    # Take top N
    final = clean_candidates[:CFG.final_output_size]

    # If we don't have enough, fill from further down the ranked list
    if len(final) < CFG.final_output_size:
        final_ids = {c for c, _ in final}
        remaining = ranked[CFG.top_n_for_pruning:]
        for cid, score in remaining:
            if len(final) >= CFG.final_output_size:
                break
            feats = candidate_features.get(cid, {})
            if feats.get("is_honeypot", 0) > 0:
                continue
            if cid not in final_ids:
                final.append((cid, score))
                final_ids.add(cid)

    final = final[:CFG.final_output_size]
    logger.info("Final: %d candidates after pruning, in %.1fs", len(final), time.time() - t4)

    # -- Stage 5: Score normalization + reasoning generation --
    logger.info("[Stage 5] Score normalization + reasoning generation...")
    t5 = time.time()

    n = len(final)
    score_span = CFG.score_range_max - CFG.score_range_min  # 0.80

    results = []
    for rank_idx, (cid, raw_score) in enumerate(final):
        rank = rank_idx + 1

        # Power curve: t^0.7 gives more spread at the top, less at the bottom
        if n > 1:
            t = rank_idx / (n - 1)  # 0.0 to 1.0
            normalized_score = round(CFG.score_range_max - score_span * (t ** CFG.score_curve_exponent), 4)
        else:
            normalized_score = CFG.score_range_max

        # Generate reasoning
        candidate = candidates_by_id.get(cid)
        if candidate:
            reasoning = generate_reasoning(
                candidate, rank, normalized_score,
                features=candidate_features.get(cid),
            )
        else:
            reasoning = f"Rank {rank} candidate."

        results.append({
            "candidate_id": cid,
            "rank": rank,
            "score": normalized_score,
            "reasoning": reasoning,
        })

    # Enforce strict monotonically non-increasing scores at 4dp precision
    for i in range(1, len(results)):
        if results[i]["score"] >= results[i - 1]["score"]:
            results[i]["score"] = round(results[i - 1]["score"] - 1e-4, 4)

    # Ensure minimum score floor
    for r in results:
        r["score"] = max(CFG.min_score_floor, r["score"])

    # Handle tie-breaking: for equal scores, sort by candidate_id ascending
    i = 0
    while i < len(results):
        j = i
        while j < len(results) and abs(results[j]["score"] - results[i]["score"]) < 1e-9:
            j += 1
        if j - i > 1:
            tied = results[i:j]
            tied.sort(key=lambda x: x["candidate_id"])
            for k, idx in enumerate(range(i, j)):
                results[idx] = tied[k]
                results[idx]["rank"] = idx + 1
        i = j

    # Re-assign ranks after tie-breaking
    for i, r in enumerate(results):
        r["rank"] = i + 1

    logger.info("Generated %d reasonings in %.1fs", len(results), time.time() - t5)

    # -- Stage 6: Write CSV --
    logger.info("[Stage 6] Writing submission to %s...", output_path)

    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for r in results:
            writer.writerow([
                r["candidate_id"],
                r["rank"],
                f"{r['score']:.4f}",
                r["reasoning"],
            ])

    # -- Stage 7: Validate --
    logger.info("[Stage 7] Validating submission...")
    errors = validate_submission(output_path)
    if errors:
        logger.warning("Validation failed (%d issues):", len(errors))
        for e in errors:
            logger.warning("  - %s", e)
    else:
        logger.info("Submission is valid.")

    # -- Summary --
    total_elapsed = time.time() - total_start
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info("Total candidates: %s", f"{n_total:,}")
    logger.info("Final ranked: %d", len(results))
    logger.info("Total time: %.1fs (%.1f min)", total_elapsed, total_elapsed / 60)
    logger.info("Output: %s", output_path)

    # Print top 10
    logger.info("Top 10 candidates:")
    for r in results[:10]:
        cand = candidates_by_id.get(r["candidate_id"], {})
        title = cand.get("profile", {}).get("current_title", "?")
        yoe = cand.get("profile", {}).get("years_of_experience", 0)
        logger.info(
            "  #%3d %s (score=%.4f) %s, %.1fy",
            r["rank"], r["candidate_id"], r["score"], title, yoe,
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Rank Redrob candidates for Senior AI Engineer JD."
    )
    parser.add_argument(
        "--candidates", default="candidates.jsonl",
        help="Path to the candidate JSONL file.",
    )
    parser.add_argument(
        "--out", default="submission.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--artifacts", default="artifacts",
        help="Directory containing pre-computed artifacts.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.candidates, args.out, args.artifacts)
