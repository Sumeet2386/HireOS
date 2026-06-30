"""
rank.py — Main inference script for the Redrob AI Candidate Ranking System.

Must run in ≤5 min, ≤16GB RAM, CPU only, no network.

Orchestrates the multi-stage pipeline:
1. Load pre-computed artifacts (FAISS index, BM25 index, features, etc.)
2. Hybrid recall (FAISS + BM25): 100K → ~5,000 unique candidates
3. Feature engineering + multi-signal scoring: ~5,000 → 300
4. Honeypot pruning: 300 → ~100 clean candidates
5. Reasoning generation: top 100 with per-candidate justifications
6. CSV output + validation

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

# Ensure ranker package is importable
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ranker.features import extract_all_features
from ranker.honeypot import detect_honeypot
from ranker.ltr import fallback_weighted_scoring
from ranker.reasoning import generate_reasoning
from ranker.validator import validate_submission


def load_artifacts(artifacts_dir: Path) -> dict:
    """Load all pre-computed artifacts from disk."""
    artifacts = {}
    t0 = time.time()

    # FAISS index
    faiss_path = artifacts_dir / "index.faiss"
    if faiss_path.exists():
        import faiss
        artifacts["faiss_index"] = faiss.read_index(str(faiss_path))
        print(f"  FAISS index: {artifacts['faiss_index'].ntotal:,} vectors")
    else:
        print("  WARNING: FAISS index not found. Skipping dense recall.")

    # BM25 index
    bm25_path = artifacts_dir / "bm25_index.pkl"
    if bm25_path.exists():
        with open(bm25_path, "rb") as f:
            artifacts["bm25_index"] = pickle.load(f)
        print("  BM25 index loaded.")
    else:
        print("  WARNING: BM25 index not found. Skipping sparse recall.")

    # JD embedding
    jd_path = artifacts_dir / "jd_embedding.npy"
    if jd_path.exists():
        artifacts["jd_embedding"] = np.load(jd_path).astype(np.float32)
        print(f"  JD embedding: shape {artifacts['jd_embedding'].shape}")

    # ID mapping
    mapping_path = artifacts_dir / "id_mapping.json"
    if mapping_path.exists():
        with open(mapping_path, "r") as f:
            artifacts["id_mapping"] = {int(k): v for k, v in json.load(f).items()}
        print(f"  ID mapping: {len(artifacts['id_mapping']):,} entries")

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
        print(f"  Pre-computed features: {len(artifacts['precomputed_features']):,} candidates, "
              f"{len(feature_cols)} features")

    elapsed = time.time() - t0
    print(f"  Artifacts loaded in {elapsed:.1f}s")

    return artifacts


def run_pipeline(
    candidates_path: str,
    output_path: str,
    artifacts_dir: str = "artifacts",
) -> None:
    """Run the full ranking pipeline."""
    total_start = time.time()
    artifacts_dir = Path(artifacts_dir)

    print("=" * 60)
    print("REDROB AI CANDIDATE RANKING SYSTEM")
    print("=" * 60)

    # ── Stage 0: Load artifacts ──
    print("\n[Stage 0] Loading artifacts...")
    artifacts = load_artifacts(artifacts_dir)

    # ── Stage 1: Load candidates and build recall set ──
    print(f"\n[Stage 1] Loading candidates from {candidates_path}...")
    t1 = time.time()

    candidates_by_id: dict[str, dict] = {}
    with open(candidates_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    # Detect format: JSON array vs JSONL
    if content.startswith("["):
        # JSON array (e.g., sample_candidates.json)
        for candidate in json.loads(content):
            cid = candidate["candidate_id"]
            candidates_by_id[cid] = candidate
    else:
        # JSONL (one JSON per line)
        for line in content.split("\n"):
            if not line.strip():
                continue
            candidate = json.loads(line)
            cid = candidate["candidate_id"]
            candidates_by_id[cid] = candidate

    n_total = len(candidates_by_id)
    print(f"  Loaded {n_total:,} candidates in {time.time()-t1:.1f}s")

    # ── Stage 2: Recall (if artifacts available) or score all ──
    print("\n[Stage 2] Candidate recall + feature extraction...")
    t2 = time.time()

    has_recall = "faiss_index" in artifacts and "jd_embedding" in artifacts
    precomputed = artifacts.get("precomputed_features", {})

    if has_recall:
        from ranker.recall import hybrid_recall
        recall_results = hybrid_recall(
            jd_embedding=artifacts["jd_embedding"],
            jd_text=(
                "senior AI engineer embeddings retrieval ranking recommendation "
                "vector database semantic search NLP transformers production "
                "Python PyTorch deployment evaluation NDCG"
            ),
            faiss_index=artifacts["faiss_index"],
            bm25_index=artifacts.get("bm25_index"),
            id_mapping=artifacts["id_mapping"],
            k_dense=5000,
            k_sparse=500 if "bm25_index" in artifacts else 0,
        )
        recalled_ids = [cid for cid, _, _ in recall_results[:5000]]
        dense_scores = {cid: ds for cid, ds, _ in recall_results}
        sparse_scores = {cid: ss for cid, _, ss in recall_results}
        print(f"  Recalled {len(recalled_ids):,} candidates via hybrid retrieval")
    else:
        # No recall artifacts — score all candidates (slower but works)
        recalled_ids = list(candidates_by_id.keys())
        dense_scores = {}
        sparse_scores = {}
        print(f"  No recall artifacts. Scoring all {len(recalled_ids):,} candidates.")

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
    # The precomputed features.parquet may lack new checks (e.g. fictional companies).
    # Run live detection on recalled candidates to update flags before scoring.
    enriched = 0
    for cid, feats in candidate_features.items():
        if cid in candidates_by_id:
            is_hp, flags = detect_honeypot(candidates_by_id[cid])
            new_hp = 1.0 if is_hp else 0.0
            new_count = float(len(flags))
            new_fictional = 1.0 if any(f.startswith("FICTIONAL_COMPANY") for f in flags) else 0.0
            new_maturity = 1.0 if any(f.startswith("MATURITY_IMPOSSIBLE") for f in flags) else 0.0

            # Update with max of precomputed vs live (never reduce flags)
            if new_hp > feats.get("is_honeypot", 0.0):
                feats["is_honeypot"] = new_hp
                enriched += 1
            feats["honeypot_flag_count"] = max(feats.get("honeypot_flag_count", 0.0), new_count)
            feats["has_fictional_company"] = max(feats.get("has_fictional_company", 0.0), new_fictional)
            feats["has_maturity_impossible"] = max(feats.get("has_maturity_impossible", 0.0), new_maturity)

    print(f"  Features extracted for {len(candidate_features):,} candidates in {time.time()-t2:.1f}s")
    if enriched > 0:
        print(f"  Live honeypot enrichment: {enriched} candidates newly flagged")

    # ── Stage 3: Multi-signal Scoring ──
    # NOTE: We use the fallback_weighted_scoring function which implements a
    # balanced multi-signal scoring formula. The LightGBM LTR model was found
    # to be dominated by saved_by_recruiters_30d (79% importance) with 17
    # critical features at zero importance. The hand-tuned formula produces
    # substantially better rankings by properly weighting title relevance,
    # skill match, semantic similarity, behavioral signals, and penalties.
    print("\n[Stage 3] Multi-signal scoring...")
    t3 = time.time()

    print("  Using multi-signal weighted scoring formula.")
    ranked = fallback_weighted_scoring(candidate_features)

    print(f"  Scored {len(ranked):,} candidates in {time.time()-t3:.1f}s")

    # ── Stage 4: Honeypot pruning ──
    # IMPORTANT: Precomputed features.parquet may not have new honeypot checks
    # (e.g., fictional company detection). Run live detection on top 300 to
    # ensure new checks are applied.
    print("\n[Stage 4] Honeypot pruning (top 300 -> top 100)...")
    t4 = time.time()

    top_300 = ranked[:300]
    clean_candidates: list[tuple[str, float]] = []
    pruned_count = 0

    for cid, score in top_300:
        feats = candidate_features.get(cid, {})

        # Always run live honeypot check if candidate data is available
        # This catches new checks (fictional companies) not in precomputed features
        if cid in candidates_by_id:
            is_hp, flags = detect_honeypot(candidates_by_id[cid])
            # Update features with live-detected flags
            feats["is_honeypot"] = 1.0 if is_hp else feats.get("is_honeypot", 0.0)
            feats["honeypot_flag_count"] = max(
                feats.get("honeypot_flag_count", 0.0),
                float(len(flags)),
            )
            feats["has_fictional_company"] = 1.0 if any(
                f.startswith("FICTIONAL_COMPANY") for f in flags
            ) else 0.0
            feats["has_maturity_impossible"] = max(
                feats.get("has_maturity_impossible", 0.0),
                1.0 if any(f.startswith("MATURITY_IMPOSSIBLE") for f in flags) else 0.0,
            )
            candidate_features[cid] = feats

        if feats.get("is_honeypot", 0) > 0:
            pruned_count += 1
            continue

        clean_candidates.append((cid, score))

    print(f"  Pruned {pruned_count} honeypots from top 300")

    # Take top 100
    final_100 = clean_candidates[:100]

    # If we don't have enough, fill from further down the ranked list
    if len(final_100) < 100:
        remaining = ranked[300:]
        for cid, score in remaining:
            if len(final_100) >= 100:
                break
            feats = candidate_features.get(cid, {})
            if feats.get("is_honeypot", 0) > 0:
                continue
            if cid not in {c for c, _ in final_100}:
                final_100.append((cid, score))

    final_100 = final_100[:100]
    print(f"  Final: {len(final_100)} candidates after pruning, in {time.time()-t4:.1f}s")

    # ── Stage 5: Score normalization + reasoning generation ──
    print("\n[Stage 5] Score normalization + reasoning generation...")
    t5 = time.time()

    # Normalize scores to a [0.20, 1.00] range using rank-proportional power curve.
    # This approach:
    #  - Gives more score spread at the top (where NDCG@10 cares most)
    #  - Eliminates tail compression (old approach had ranks 77-100 all at ~0.1976-0.2000)
    #  - Guarantees unique scores at 4 decimal places for 100 ranks
    #  - Preserves monotonicity by construction (no post-hoc fixing needed)
    n = len(final_100)

    results = []
    for rank_idx, (cid, raw_score) in enumerate(final_100):
        rank = rank_idx + 1

        # Power curve: t^0.7 gives more spread at the top, less at the bottom
        # Rank 1 → 1.0, Rank 100 → 0.20
        if n > 1:
            t = rank_idx / (n - 1)  # 0.0 to 1.0
            normalized_score = round(1.0 - 0.80 * (t ** 0.7), 4)
        else:
            normalized_score = 1.0

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

    # Ensure minimum score is at least 0.01
    for r in results:
        r["score"] = max(0.01, r["score"])

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

    print(f"  Generated {len(results)} reasonings in {time.time()-t5:.1f}s")

    # ── Stage 6: Write CSV ──
    print(f"\n[Stage 6] Writing submission to {output_path}...")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for r in results:
            writer.writerow([
                r["candidate_id"],
                r["rank"],
                f"{r['score']:.4f}",
                r["reasoning"],
            ])

    # ── Stage 7: Validate ──
    print("\n[Stage 7] Validating submission...")
    errors = validate_submission(output_path)
    if errors:
        print(f"  WARNING: Validation failed ({len(errors)} issues):")
        for e in errors:
            print(f"    - {e}")
    else:
        print("  OK: Submission is valid!")

    # ── Summary ──
    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Total candidates: {n_total:,}")
    print(f"Final ranked: {len(results)}")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"Output: {output_path}")

    # Print top 10
    print(f"\nTop 10 candidates:")
    for r in results[:10]:
        cand = candidates_by_id.get(r["candidate_id"], {})
        title = cand.get("profile", {}).get("current_title", "?")
        yoe = cand.get("profile", {}).get("years_of_experience", 0)
        print(f"  #{r['rank']:>3} {r['candidate_id']} (score={r['score']:.4f}) "
              f"{title}, {yoe:.1f}y")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank Redrob candidates for Senior AI Engineer JD."
    )
    parser.add_argument(
        "--candidates", default="candidates.jsonl",
        help="Path to the candidate JSONL file."
    )
    parser.add_argument(
        "--out", default="submission.csv",
        help="Output CSV path."
    )
    parser.add_argument(
        "--artifacts", default="artifacts",
        help="Directory containing pre-computed artifacts."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.candidates, args.out, args.artifacts)
