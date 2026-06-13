"""
evaluate.py — Comprehensive Evaluation & QA for the Redrob Ranking Pipeline.

Runs every check we can think of and prints a detailed metrics report.
"""

from __future__ import annotations

import csv
import json
import os
import pickle
import re
import sys
import time
import tracemalloc
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ranker.constants import (
    CORE_AI_SKILLS,
    HIGH_SIGNAL_SKILLS,
    HIGH_SIGNAL_TITLES,
    ADJACENT_SIGNAL_TITLES,
    TIER1_INDIA_CITIES,
    CONSULTING_FIRMS,
    NEGATIVE_TITLE_PATTERNS,
)
from ranker.features import extract_all_features
from ranker.honeypot import detect_honeypot
from ranker.validator import validate_submission


CANDIDATES_PATH = (
    r"d:\Hackathons\Data Challenge IndiaRuns"
    r"\[PUB] India_runs_data_and_ai_challenge"
    r"\[PUB] India_runs_data_and_ai_challenge"
    r"\India_runs_data_and_ai_challenge\candidates.jsonl"
)
SUBMISSION_CSV = ROOT / "final_submission.csv"
ARTIFACTS_DIR = ROOT / "artifacts"

SEP = "=" * 70


def load_candidates(path: str) -> dict[str, dict]:
    candidates = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            candidates[c["candidate_id"]] = c
    return candidates


def load_submission(csv_path: Path) -> list[dict]:
    rows = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "candidate_id": row["candidate_id"].strip(),
                "rank": int(row["rank"]),
                "score": float(row["score"]),
                "reasoning": row["reasoning"].strip(),
            })
    return rows


def main():
    print(SEP)
    print("REDROB RANKING SYSTEM — COMPREHENSIVE EVALUATION")
    print(SEP)

    # ────────────────────────────────────────────────────────────────────────
    # 1. CSV FORMAT VALIDATION
    # ────────────────────────────────────────────────────────────────────────
    print("\n[1/10] CSV FORMAT VALIDATION")
    print("-" * 40)
    errors = validate_submission(str(SUBMISSION_CSV))
    if errors:
        for e in errors:
            print(f"  FAIL: {e}")
        print(f"  RESULT: FAILED ({len(errors)} errors)")
    else:
        print("  Columns: candidate_id, rank, score, reasoning")
        print("  Row count: 100 data rows + 1 header")
        print("  Candidate ID format: CAND_XXXXXXX -- all valid")
        print("  Ranks: 1-100, unique, sequential")
        print("  Scores: monotonically non-increasing")
        print("  Tie-breaking: candidate_id ascending for equal scores")
        print("  RESULT: ALL CHECKS PASSED")

    # ────────────────────────────────────────────────────────────────────────
    # 2. LOAD DATA
    # ────────────────────────────────────────────────────────────────────────
    print(f"\n[2/10] LOADING DATA")
    print("-" * 40)
    t0 = time.time()
    all_candidates = load_candidates(CANDIDATES_PATH)
    print(f"  Total candidates in dataset: {len(all_candidates):,}")
    submission = load_submission(SUBMISSION_CSV)
    print(f"  Candidates in submission: {len(submission)}")
    print(f"  Load time: {time.time()-t0:.1f}s")

    # ────────────────────────────────────────────────────────────────────────
    # 3. SCORE DISTRIBUTION
    # ────────────────────────────────────────────────────────────────────────
    print(f"\n[3/10] SCORE DISTRIBUTION & INTEGRITY")
    print("-" * 40)
    scores = [r["score"] for r in submission]
    print(f"  Min score:  {min(scores):.4f}")
    print(f"  Max score:  {max(scores):.4f}")
    print(f"  Mean score: {np.mean(scores):.4f}")
    print(f"  Std score:  {np.std(scores):.4f}")
    print(f"  Median:     {np.median(scores):.4f}")

    # Score buckets
    buckets = {"0.8-1.0": 0, "0.6-0.8": 0, "0.4-0.6": 0, "0.2-0.4": 0, "0.0-0.2": 0}
    for s in scores:
        if s >= 0.8:
            buckets["0.8-1.0"] += 1
        elif s >= 0.6:
            buckets["0.6-0.8"] += 1
        elif s >= 0.4:
            buckets["0.4-0.6"] += 1
        elif s >= 0.2:
            buckets["0.2-0.4"] += 1
        else:
            buckets["0.0-0.2"] += 1
    print(f"\n  Score bucket distribution:")
    for bucket, count in buckets.items():
        bar = "#" * count
        print(f"    [{bucket}]: {count:>3} {bar}")

    # Monotonicity check
    is_monotonic = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
    print(f"\n  Monotonically non-increasing: {'PASS' if is_monotonic else 'FAIL'}")

    # Unique candidate check
    cids = [r["candidate_id"] for r in submission]
    print(f"  Unique candidate IDs: {len(set(cids))} / {len(cids)} -- {'PASS' if len(set(cids)) == len(cids) else 'FAIL'}")

    # ────────────────────────────────────────────────────────────────────────
    # 4. TITLE ANALYSIS
    # ────────────────────────────────────────────────────────────────────────
    print(f"\n[4/10] TITLE RELEVANCE ANALYSIS")
    print("-" * 40)

    title_counter = Counter()
    title_relevance_scores = []
    for r in submission:
        cand = all_candidates.get(r["candidate_id"], {})
        title = cand.get("profile", {}).get("current_title", "Unknown")
        title_counter[title] += 1
        tl = title.lower()
        if any(t in tl for t in HIGH_SIGNAL_TITLES):
            title_relevance_scores.append(1.0)
        elif any(t in tl for t in ADJACENT_SIGNAL_TITLES):
            title_relevance_scores.append(0.7)
        else:
            title_relevance_scores.append(0.3)

    high_signal = sum(1 for s in title_relevance_scores if s == 1.0)
    adjacent = sum(1 for s in title_relevance_scores if s == 0.7)
    other = sum(1 for s in title_relevance_scores if s < 0.7)
    print(f"  High-signal AI/ML titles: {high_signal}/100 ({high_signal}%)")
    print(f"  Adjacent signal titles:   {adjacent}/100 ({adjacent}%)")
    print(f"  Other titles:             {other}/100 ({other}%)")
    print(f"  Mean title relevance:     {np.mean(title_relevance_scores):.3f}")

    print(f"\n  Top 10 title distribution:")
    for title, count in title_counter.most_common(10):
        print(f"    {title}: {count}")

    # Top 10 title check
    print(f"\n  Top 10 candidates' titles:")
    for r in submission[:10]:
        cand = all_candidates.get(r["candidate_id"], {})
        title = cand.get("profile", {}).get("current_title", "?")
        yoe = cand.get("profile", {}).get("years_of_experience", 0)
        print(f"    #{r['rank']:>2} {r['candidate_id']} | {title} | {yoe:.1f}y | score={r['score']:.4f}")

    # ────────────────────────────────────────────────────────────────────────
    # 5. YOE DISTRIBUTION
    # ────────────────────────────────────────────────────────────────────────
    print(f"\n[5/10] YEARS OF EXPERIENCE DISTRIBUTION")
    print("-" * 40)

    yoe_values = []
    for r in submission:
        cand = all_candidates.get(r["candidate_id"], {})
        yoe = cand.get("profile", {}).get("years_of_experience", 0)
        yoe_values.append(yoe)

    in_ideal = sum(1 for y in yoe_values if 5 <= y <= 9)
    in_acceptable = sum(1 for y in yoe_values if 4 <= y <= 10)
    print(f"  Min YoE:           {min(yoe_values):.1f}")
    print(f"  Max YoE:           {max(yoe_values):.1f}")
    print(f"  Mean YoE:          {np.mean(yoe_values):.1f}")
    print(f"  Median YoE:        {np.median(yoe_values):.1f}")
    print(f"  Std YoE:           {np.std(yoe_values):.1f}")
    print(f"  In ideal range (5-9y):     {in_ideal}/100 ({in_ideal}%)")
    print(f"  In acceptable range (4-10y): {in_acceptable}/100 ({in_acceptable}%)")

    # YoE buckets
    yoe_buckets = Counter()
    for y in yoe_values:
        if y < 3:
            yoe_buckets["<3y"] += 1
        elif y < 5:
            yoe_buckets["3-5y"] += 1
        elif y < 7:
            yoe_buckets["5-7y"] += 1
        elif y < 9:
            yoe_buckets["7-9y"] += 1
        else:
            yoe_buckets["9y+"] += 1
    print(f"\n  YoE bucket distribution:")
    for bucket in ["<3y", "3-5y", "5-7y", "7-9y", "9y+"]:
        count = yoe_buckets.get(bucket, 0)
        bar = "#" * count
        print(f"    [{bucket:>4}]: {count:>3} {bar}")

    # ────────────────────────────────────────────────────────────────────────
    # 6. LOCATION ANALYSIS
    # ────────────────────────────────────────────────────────────────────────
    print(f"\n[6/10] LOCATION ANALYSIS")
    print("-" * 40)

    india_count = 0
    tier1_count = 0
    outside_india = []
    for r in submission:
        cand = all_candidates.get(r["candidate_id"], {})
        profile = cand.get("profile", {})
        country = (profile.get("country") or "").lower()
        location = (profile.get("location") or "").lower()
        if country == "india":
            india_count += 1
            if any(city in location for city in TIER1_INDIA_CITIES):
                tier1_count += 1
        else:
            outside_india.append((r["rank"], r["candidate_id"], profile.get("location", "?")))

    print(f"  India-based:       {india_count}/100 ({india_count}%)")
    print(f"  Tier-1 India:      {tier1_count}/100 ({tier1_count}%)")
    print(f"  Outside India:     {len(outside_india)}/100 ({len(outside_india)}%)")
    if outside_india:
        print(f"\n  Non-India candidates:")
        for rank, cid, loc in outside_india[:10]:
            print(f"    #{rank} {cid}: {loc}")

    # ────────────────────────────────────────────────────────────────────────
    # 7. HONEYPOT ANALYSIS ON FULL DATASET
    # ────────────────────────────────────────────────────────────────────────
    print(f"\n[7/10] HONEYPOT ANALYSIS")
    print("-" * 40)

    # Check if any submission candidates are honeypots (they shouldn't be)
    honeypot_in_submission = 0
    check_categories = Counter()
    for r in submission:
        cand = all_candidates.get(r["candidate_id"])
        if cand:
            is_hp, flags = detect_honeypot(cand)
            if is_hp:
                honeypot_in_submission += 1
                print(f"  WARNING: Honeypot in submission! #{r['rank']} {r['candidate_id']}")
                for f in flags:
                    print(f"    - {f}")
            for f in flags:
                cat = f.split(":")[0]
                check_categories[cat] += 1

    print(f"  Honeypots in submission: {honeypot_in_submission}/100 -- {'PASS (0 found)' if honeypot_in_submission == 0 else 'FAIL'}")

    # Load precomputed features to count overall honeypots
    features_path = ARTIFACTS_DIR / "features.parquet"
    if features_path.exists():
        import pyarrow.parquet as pq
        feat_table = pq.read_table(features_path)
        feat_dict = feat_table.to_pydict()
        total_honeypots = sum(1 for v in feat_dict.get("is_honeypot", []) if v > 0)
        print(f"  Total honeypots in dataset: {total_honeypots:,}/100,000 ({total_honeypots/1000:.1f}%)")

    # ────────────────────────────────────────────────────────────────────────
    # 8. REASONING QUALITY CHECK
    # ────────────────────────────────────────────────────────────────────────
    print(f"\n[8/10] REASONING QUALITY CHECK")
    print("-" * 40)

    reasoning_lengths = []
    has_title_mention = 0
    has_yoe_mention = 0
    has_skills_mention = 0
    has_considerations = 0
    empty_reasoning = 0

    for r in submission:
        reasoning = r["reasoning"]
        reasoning_lengths.append(len(reasoning))

        if not reasoning or len(reasoning) < 10:
            empty_reasoning += 1
            continue

        cand = all_candidates.get(r["candidate_id"], {})
        title = cand.get("profile", {}).get("current_title", "")

        # Check if title is mentioned
        if title.lower() in reasoning.lower() or title in reasoning:
            has_title_mention += 1

        # Check YoE mention
        if "year" in reasoning.lower():
            has_yoe_mention += 1

        # Check skills mention
        if "expertise" in reasoning.lower() or "skill" in reasoning.lower():
            has_skills_mention += 1

        # Check considerations
        if "consideration" in reasoning.lower() or "concern" in reasoning.lower() or "No significant" in reasoning:
            has_considerations += 1

    print(f"  Empty/trivial reasonings:  {empty_reasoning}/100 -- {'PASS' if empty_reasoning == 0 else 'FAIL'}")
    print(f"  Contains candidate title:  {has_title_mention}/100")
    print(f"  Contains YoE reference:    {has_yoe_mention}/100")
    print(f"  Contains skills reference: {has_skills_mention}/100")
    print(f"  Contains risk assessment:  {has_considerations}/100")
    print(f"  Min reasoning length:      {min(reasoning_lengths)} chars")
    print(f"  Max reasoning length:      {max(reasoning_lengths)} chars")
    print(f"  Mean reasoning length:     {np.mean(reasoning_lengths):.0f} chars")

    # Sample reasonings
    print(f"\n  Sample reasonings (ranks 1, 50, 100):")
    for idx in [0, 49, 99]:
        r = submission[idx]
        print(f"    #{r['rank']}: \"{r['reasoning'][:120]}...\"")

    # ────────────────────────────────────────────────────────────────────────
    # 9. LTR MODEL & FEATURE IMPORTANCE
    # ────────────────────────────────────────────────────────────────────────
    print(f"\n[9/10] LTR MODEL & FEATURE ANALYSIS")
    print("-" * 40)

    # Feature importance
    fi_path = ARTIFACTS_DIR / "feature_importance.json"
    if fi_path.exists():
        with open(fi_path) as f:
            feat_importance_raw = json.load(f)
        # Handle both dict and list-of-lists formats
        if isinstance(feat_importance_raw, dict):
            sorted_fi = sorted(feat_importance_raw.items(), key=lambda x: -x[1])
        else:
            sorted_fi = sorted(feat_importance_raw, key=lambda x: -x[1])
        print(f"  Top 15 features by importance (gain):")
        for name, gain in sorted_fi[:15]:
            bar = "#" * max(1, int(gain / 5))
            print(f"    {name:40s} {gain:8.1f}  {bar}")

        total_gain = sum(v for _, v in sorted_fi)
        top5_gain = sum(v for _, v in sorted_fi[:5])
        print(f"\n  Total importance (all features): {total_gain:.1f}")
        print(f"  Top 5 features share:           {top5_gain/total_gain*100:.1f}%")
        zero_importance = sum(1 for _, v in sorted_fi if v == 0)
        print(f"  Zero-importance features:       {zero_importance}/{len(sorted_fi)}")

    # Weak labels stats
    wl_path = ARTIFACTS_DIR / "weak_labels.json"
    if wl_path.exists():
        with open(wl_path) as f:
            weak_labels = json.load(f)
        # Detect the score key
        score_key = "weak_label" if "weak_label" in weak_labels[0] else "score"
        label_scores = [entry[score_key] for entry in weak_labels]
        print(f"\n  Weak label statistics:")
        print(f"    Total labeled candidates: {len(weak_labels):,}")
        print(f"    Score range: [{min(label_scores)}, {max(label_scores)}]")
        print(f"    Mean: {np.mean(label_scores):.2f}, Std: {np.std(label_scores):.2f}")
        label_dist = Counter()
        for s in label_scores:
            if s <= 1:
                label_dist["0-1"] += 1
            elif s <= 3:
                label_dist["2-3"] += 1
            elif s <= 5:
                label_dist["4-5"] += 1
            elif s <= 7:
                label_dist["6-7"] += 1
            elif s <= 9:
                label_dist["8-9"] += 1
            else:
                label_dist["10"] += 1
        print(f"    Distribution:")
        for bucket in ["0-1", "2-3", "4-5", "6-7", "8-9", "10"]:
            count = label_dist.get(bucket, 0)
            print(f"      [{bucket}]: {count}")

    # ────────────────────────────────────────────────────────────────────────
    # 10. RUNTIME & MEMORY PROFILING
    # ────────────────────────────────────────────────────────────────────────
    print(f"\n[10/10] RUNTIME & MEMORY PROFILING")
    print("-" * 40)

    # Measure artifact sizes
    total_artifact_mb = 0.0
    print(f"  Artifact sizes:")
    for p in sorted(ARTIFACTS_DIR.iterdir()):
        size_mb = p.stat().st_size / (1024 * 1024)
        total_artifact_mb += size_mb
        print(f"    {p.name:35s} {size_mb:8.1f} MB")
    print(f"    {'TOTAL':35s} {total_artifact_mb:8.1f} MB")

    # Memory-profiled pipeline run
    print(f"\n  Running full pipeline with memory tracking...")
    tracemalloc.start()
    t_start = time.time()

    # Load artifacts
    import faiss
    import pyarrow.parquet as pq
    import lightgbm as lgb

    faiss_index = faiss.read_index(str(ARTIFACTS_DIR / "index.faiss"))
    with open(ARTIFACTS_DIR / "bm25_index.pkl", "rb") as f:
        bm25_index = pickle.load(f)
    jd_embedding = np.load(ARTIFACTS_DIR / "jd_embedding.npy").astype(np.float32)
    with open(ARTIFACTS_DIR / "id_mapping.json") as f:
        id_mapping = {int(k): v for k, v in json.load(f).items()}
    feat_table = pq.read_table(ARTIFACTS_DIR / "features.parquet")
    feat_pydict = feat_table.to_pydict()
    feat_cids = feat_pydict.pop("candidate_id")
    feat_cols = list(feat_pydict.keys())
    precomputed = {
        cid: {col: feat_pydict[col][i] for col in feat_cols}
        for i, cid in enumerate(feat_cids)
    }
    ltr_model = lgb.Booster(model_file=str(ARTIFACTS_DIR / "lgbm_ltr_model.bin"))
    with open(ARTIFACTS_DIR / "feature_columns.json") as f:
        feature_columns = json.load(f)

    t_artifacts = time.time() - t_start

    # Load candidates
    t_load = time.time()
    candidates_by_id = load_candidates(CANDIDATES_PATH)
    t_load = time.time() - t_load

    # Recall
    t_recall = time.time()
    from ranker.recall import hybrid_recall
    recall_results = hybrid_recall(
        jd_embedding=jd_embedding,
        jd_text="senior AI engineer embeddings retrieval ranking recommendation vector database semantic search NLP transformers production Python PyTorch deployment evaluation NDCG",
        faiss_index=faiss_index,
        bm25_index=bm25_index,
        id_mapping=id_mapping,
        k_dense=5000,
        k_sparse=500,
    )
    recalled_ids = [cid for cid, _, _ in recall_results[:5000]]
    dense_scores = {cid: ds for cid, ds, _ in recall_results}
    sparse_scores = {cid: ss for cid, _, ss in recall_results}
    t_recall = time.time() - t_recall

    # Feature extraction
    t_feat = time.time()
    candidate_features = {}
    for cid in recalled_ids:
        if cid in precomputed:
            features = dict(precomputed[cid])
        elif cid in candidates_by_id:
            features = extract_all_features(candidates_by_id[cid])
        else:
            continue
        features["cosine_similarity_jd"] = dense_scores.get(cid, 0.0)
        features["bm25_score_jd"] = sparse_scores.get(cid, 0.0)
        candidate_features[cid] = features
    t_feat = time.time() - t_feat

    # LTR ranking
    t_ltr = time.time()
    from ranker.ltr import rank_candidates as ltr_rank
    ranked = ltr_rank(ltr_model, candidate_features, feature_columns)
    t_ltr = time.time() - t_ltr

    # Honeypot pruning
    t_prune = time.time()
    top_300 = ranked[:300]
    clean = []
    pruned = 0
    for cid, score in top_300:
        feats = candidate_features.get(cid, {})
        if feats.get("is_honeypot", 0) > 0:
            pruned += 1
            continue
        clean.append((cid, score))
    final_100 = clean[:100]
    t_prune = time.time() - t_prune

    # Reasoning
    t_reason = time.time()
    from ranker.reasoning import generate_reasoning
    for rank_idx, (cid, _) in enumerate(final_100):
        cand = candidates_by_id.get(cid)
        if cand:
            generate_reasoning(cand, rank_idx + 1, 0.5, features=candidate_features.get(cid))
    t_reason = time.time() - t_reason

    total_time = time.time() - t_start
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\n  Stage timing breakdown:")
    print(f"    Artifact loading:    {t_artifacts:6.2f}s")
    print(f"    Candidate loading:   {t_load:6.2f}s")
    print(f"    Hybrid recall:       {t_recall:6.2f}s")
    print(f"    Feature extraction:  {t_feat:6.2f}s")
    print(f"    LTR re-ranking:      {t_ltr:6.2f}s")
    print(f"    Honeypot pruning:    {t_prune:6.2f}s")
    print(f"    Reasoning gen:       {t_reason:6.2f}s")
    print(f"    {'TOTAL':25s} {total_time:6.2f}s")

    print(f"\n  Memory usage:")
    print(f"    Current:  {current_mem / 1024 / 1024:.1f} MB")
    print(f"    Peak:     {peak_mem / 1024 / 1024:.1f} MB")
    under_limit = peak_mem < 16 * 1024 * 1024 * 1024
    under_time = total_time < 300
    print(f"\n  Runtime < 5 min:   {'PASS' if under_time else 'FAIL'} ({total_time:.1f}s / 300s)")
    print(f"  Memory < 16 GB:    {'PASS' if under_limit else 'FAIL'} ({peak_mem / 1024 / 1024:.1f} MB / 16384 MB)")
    print(f"  Recall candidates: {len(recalled_ids):,}")
    print(f"  Honeypots pruned:  {pruned}")
    print(f"  Final output:      {len(final_100)}")

    # ────────────────────────────────────────────────────────────────────────
    # FINAL SUMMARY
    # ────────────────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("FINAL EVALUATION SUMMARY")
    print(SEP)

    checks = [
        ("CSV format validation", len(errors) == 0),
        ("100 candidates output", len(submission) == 100),
        ("Scores monotonically non-increasing", is_monotonic),
        ("Unique candidate IDs", len(set(cids)) == len(cids)),
        ("No honeypots in final output", honeypot_in_submission == 0),
        ("No empty reasonings", empty_reasoning == 0),
        ("Runtime < 5 min (300s)", under_time),
        ("Memory < 16 GB", under_limit),
        (f"YoE in ideal range (5-9y) >= 70%", in_ideal >= 70),
        (f"High-signal titles >= 40%", high_signal >= 40),
    ]

    all_pass = True
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        icon = "[OK]" if passed else "[XX]"
        print(f"  {icon} {name}: {status}")
        if not passed:
            all_pass = False

    print(f"\n  {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
    print(SEP)


if __name__ == "__main__":
    main()
