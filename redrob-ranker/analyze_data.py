"""Deep analysis script to understand ground truth signals before making any changes."""

import json
import csv
import math
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
CANDIDATES_PATH = ROOT.parent / "India_runs_data_and_ai_challenge" / "candidates.jsonl"

SEP = "=" * 70

# ============================================================================
# 1. WEAK LABELS ANALYSIS
# ============================================================================
print(SEP)
print("1. WEAK LABELS (GPT-4o-mini ground truth proxy)")
print(SEP)

with open(ARTIFACTS / "weak_labels.json") as f:
    wl = json.load(f)

scores = [w["weak_label"] for w in wl]
print(f"Total labeled: {len(wl)}")
print(f"Mean: {np.mean(scores):.2f}, Median: {np.median(scores):.1f}, Std: {np.std(scores):.2f}")
print(f"\nDistribution:")
for bucket in range(0, 11):
    count = sum(1 for s in scores if int(s) == bucket)
    bar = "#" * (count // 2)
    print(f"  Score {bucket:>2}: {count:>4}  {bar}")

# High scorers
high = [w for w in wl if w["weak_label"] >= 8]
print(f"\nHigh scorers (>=8): {len(high)}")
for h in sorted(high, key=lambda x: -x["weak_label"])[:15]:
    cid = h["candidate_id"]
    score = h["weak_label"]
    reason = h.get("reason", "")[:120]
    print(f"  {cid} score={score} reason={reason}")

# ============================================================================
# 2. FEATURE IMPORTANCE ANALYSIS
# ============================================================================
print(f"\n{SEP}")
print("2. LIGHTGBM FEATURE IMPORTANCE")
print(SEP)

with open(ARTIFACTS / "feature_importance.json") as f:
    fi = json.load(f)

if isinstance(fi, dict):
    sorted_fi = sorted(fi.items(), key=lambda x: -x[1])
else:
    sorted_fi = sorted(fi, key=lambda x: -x[1])

total = sum(v for _, v in sorted_fi)
print(f"Total importance: {total:.1f}")
print(f"\nAll features by importance:")
cumulative = 0
for name, gain in sorted_fi:
    pct = gain / total * 100
    cumulative += pct
    bar = "#" * max(1, int(pct / 2))
    print(f"  {name:45s} {gain:8.1f} ({pct:5.1f}%) [cum: {cumulative:5.1f}%]  {bar}")

zero_count = sum(1 for _, v in sorted_fi if v == 0)
print(f"\nZero importance: {zero_count}/{len(sorted_fi)}")

# ============================================================================
# 3. LOAD FEATURES AND ANALYZE CORRELATIONS WITH WEAK LABELS
# ============================================================================
print(f"\n{SEP}")
print("3. FEATURE-TO-WEAK-LABEL CORRELATIONS")
print(SEP)

# Load features
feat_table = pq.read_table(ARTIFACTS / "features.parquet")
feat_dict = feat_table.to_pydict()
feat_cids = feat_dict.pop("candidate_id")
feat_cols = list(feat_dict.keys())

# Build feature lookup
feat_lookup = {}
for i, cid in enumerate(feat_cids):
    feat_lookup[cid] = {col: feat_dict[col][i] for col in feat_cols}

# Match weak labels to features
wl_cids_with_features = []
wl_scores_matched = []
wl_features_matched = defaultdict(list)

for w in wl:
    cid = w["candidate_id"]
    if cid in feat_lookup:
        wl_cids_with_features.append(cid)
        wl_scores_matched.append(w["weak_label"])
        for col in feat_cols:
            wl_features_matched[col].append(feat_lookup[cid][col])

print(f"Weak labels matched to features: {len(wl_cids_with_features)}/{len(wl)}")

# Compute correlations
correlations = []
for col in feat_cols:
    vals = np.array(wl_features_matched[col])
    labels = np.array(wl_scores_matched)
    
    # Skip constant features
    if np.std(vals) < 1e-9 or np.std(labels) < 1e-9:
        correlations.append((col, 0.0))
        continue
    
    corr = np.corrcoef(vals, labels)[0, 1]
    correlations.append((col, corr if not np.isnan(corr) else 0.0))

correlations.sort(key=lambda x: -abs(x[1]))

print(f"\nFeature correlations with weak labels (sorted by |r|):")
for name, corr in correlations:
    direction = "+" if corr > 0 else "-"
    bar_len = int(abs(corr) * 40)
    bar = "#" * bar_len
    print(f"  {name:45s}  r={corr:+.4f}  {bar}")

# ============================================================================
# 4. CROSS-CHECK: OUR SUBMISSION vs WEAK LABELS
# ============================================================================
print(f"\n{SEP}")
print("4. OUR SUBMISSION vs WEAK LABELS")
print(SEP)

# Load submission
submission_cids = []
with open(ROOT / "final_submission.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        submission_cids.append(row["candidate_id"].strip())

# How many of our top 100 have weak labels?
wl_lookup = {w["candidate_id"]: w["weak_label"] for w in wl}

in_submission_with_wl = []
for i, cid in enumerate(submission_cids):
    if cid in wl_lookup:
        in_submission_with_wl.append((i + 1, cid, wl_lookup[cid]))

print(f"Our submission candidates with weak labels: {len(in_submission_with_wl)}/100")
if in_submission_with_wl:
    wl_scores_in_sub = [s for _, _, s in in_submission_with_wl]
    print(f"Their avg weak label: {np.mean(wl_scores_in_sub):.2f}")
    print(f"\nMatched candidates (rank, id, weak_label):")
    for rank, cid, score in in_submission_with_wl:
        print(f"  #{rank:>3} {cid} weak_label={score:.0f}")

# What's the avg weak label of the BEST candidates not in our submission?
not_in_submission = [(cid, score) for cid, score in wl_lookup.items() if cid not in set(submission_cids)]
not_in_submission.sort(key=lambda x: -x[1])
print(f"\nBest candidates NOT in our submission (by weak label):")
for cid, score in not_in_submission[:20]:
    feats = feat_lookup.get(cid, {})
    title_rel = feats.get("current_title_relevance", -1)
    entail = feats.get("skill_text_entailment_rate", -1)
    is_hp = feats.get("is_honeypot", -1)
    saved = feats.get("saved_by_recruiters_30d", -1)
    yoe = feats.get("years_of_experience", -1)
    print(f"  {cid} wl={score:.0f} title_rel={title_rel:.2f} entail={entail:.2f} hp={is_hp:.0f} saved={saved:.0f} yoe={yoe:.1f}")

# ============================================================================
# 5. ANALYZE BEHAVIORAL SIGNALS IN HIGH vs LOW WEAK-LABEL CANDIDATES
# ============================================================================
print(f"\n{SEP}")
print("5. BEHAVIORAL SIGNALS: HIGH vs LOW WEAK LABEL CANDIDATES")
print(SEP)

high_wl = [w for w in wl if w["weak_label"] >= 7 and w["candidate_id"] in feat_lookup]
low_wl = [w for w in wl if w["weak_label"] <= 3 and w["candidate_id"] in feat_lookup]

behavioral_features = [
    "saved_by_recruiters_30d", "search_appearance_30d", "recruiter_response_rate",
    "activity_decay_score", "github_activity_score", "profile_views_30d",
    "applications_30d", "open_to_work", "notice_period_days",
    "interview_completion_rate", "verification_score",
]

print(f"High weak label (>=7): {len(high_wl)} candidates")
print(f"Low weak label  (<=3): {len(low_wl)} candidates")
print(f"\n{'Feature':45s} {'High(>=7)':>12} {'Low(<=3)':>12}  {'Delta':>8}  {'Signal?'}")

for feat in behavioral_features:
    high_vals = [feat_lookup[w["candidate_id"]][feat] for w in high_wl if feat in feat_lookup[w["candidate_id"]]]
    low_vals = [feat_lookup[w["candidate_id"]][feat] for w in low_wl if feat in feat_lookup[w["candidate_id"]]]
    
    if high_vals and low_vals:
        high_mean = np.mean(high_vals)
        low_mean = np.mean(low_vals)
        delta = high_mean - low_mean
        signal = "YES" if abs(delta) > 0.05 * max(abs(high_mean), abs(low_mean), 1) else "no"
        print(f"  {feat:45s} {high_mean:>12.3f} {low_mean:>12.3f}  {delta:>+8.3f}  {signal}")

# Also check structural features
structural_features = [
    "current_title_relevance", "best_title_relevance", "skill_text_entailment_rate",
    "num_core_ai_skills", "num_high_signal_skills", "skill_proficiency_score",
    "years_of_experience", "has_product_company_exp", "all_consulting_career",
    "is_india", "is_tier1_india", "is_honeypot", "honeypot_flag_count",
]

print(f"\n{'Structural Feature':45s} {'High(>=7)':>12} {'Low(<=3)':>12}  {'Delta':>8}  {'Signal?'}")

for feat in structural_features:
    high_vals = [feat_lookup[w["candidate_id"]][feat] for w in high_wl if feat in feat_lookup[w["candidate_id"]]]
    low_vals = [feat_lookup[w["candidate_id"]][feat] for w in low_wl if feat in feat_lookup[w["candidate_id"]]]
    
    if high_vals and low_vals:
        high_mean = np.mean(high_vals)
        low_mean = np.mean(low_vals)
        delta = high_mean - low_mean
        signal = "YES" if abs(delta) > 0.05 * max(abs(high_mean), abs(low_mean), 1) else "no"
        print(f"  {feat:45s} {high_mean:>12.3f} {low_mean:>12.3f}  {delta:>+8.3f}  {signal}")

# ============================================================================
# 6. SIMULATE NDCG IMPACT: What if we use weak labels as ground truth?
# ============================================================================
print(f"\n{SEP}")
print("6. SIMULATED NDCG WITH WEAK LABELS AS GROUND TRUTH")
print(SEP)

# We can only compute this for candidates that have both a submission rank AND a weak label
# This is a PROXY — the real ground truth covers all 100K candidates

def dcg_at_k(relevances, k):
    """Compute DCG@k."""
    rel = relevances[:k]
    return sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(rel))

def ndcg_at_k(relevances, ideal_relevances, k):
    """Compute NDCG@k."""
    actual_dcg = dcg_at_k(relevances, k)
    ideal_dcg = dcg_at_k(sorted(ideal_relevances, reverse=True), k)
    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg

# Build relevance vector for our submission (using weak labels where available, 5.0 default)
our_relevances = []
for cid in submission_cids:
    our_relevances.append(wl_lookup.get(cid, 5.0))  # neutral default for unlabeled

# Build ideal relevance (best possible from all weak-labeled candidates)
all_wl_scores = sorted(wl_lookup.values(), reverse=True)

print(f"Our submission relevances (from weak labels, 5.0 for unlabeled):")
print(f"  Top 10 relevances: {[our_relevances[i] for i in range(10)]}")
print(f"  Top 50 avg: {np.mean(our_relevances[:50]):.2f}")
print(f"  Full 100 avg: {np.mean(our_relevances):.2f}")

print(f"\nIdeal relevances (best weak-labeled candidates globally):")
print(f"  Top 10: {all_wl_scores[:10]}")
print(f"  Top 50 avg: {np.mean(all_wl_scores[:50]):.2f}")

# NDCG computation
for k in [10, 50, 100]:
    ndcg = ndcg_at_k(our_relevances, all_wl_scores[:100], k)
    print(f"  NDCG@{k}: {ndcg:.4f}")

composite = 0.50 * ndcg_at_k(our_relevances, all_wl_scores[:100], 10) + \
            0.30 * ndcg_at_k(our_relevances, all_wl_scores[:100], 50) + \
            0.15 * ndcg_at_k(our_relevances, all_wl_scores[:100], 100) + \
            0.05 * (sum(1 for r in our_relevances[:10] if r >= 7) / 10)
print(f"\n  Estimated composite: {composite:.4f}")

# ============================================================================
# 7. ENTAILMENT ANALYSIS: Are the 15 missing high-quality candidates real?
# ============================================================================
print(f"\n{SEP}")
print("7. ENTAILMENT RATE ANALYSIS: WHO ARE WE PENALIZING?")
print(SEP)

# Find all candidates with low entailment but high other scores
low_entail_high_quality = []
for cid, feats in feat_lookup.items():
    entail = feats.get("skill_text_entailment_rate", 1.0)
    title_rel = feats.get("current_title_relevance", 0)
    skill_count = feats.get("num_high_signal_skills", 0)
    yoe = feats.get("years_of_experience", 0)
    is_hp = feats.get("is_honeypot", 0)
    
    if entail < 0.15 and title_rel >= 0.7 and skill_count >= 3 and 4 <= yoe <= 10 and is_hp == 0:
        wl_score = wl_lookup.get(cid, None)
        low_entail_high_quality.append((cid, entail, title_rel, skill_count, yoe, wl_score))

print(f"Candidates with LOW entailment (<0.15) but HIGH surface quality:")
print(f"  (title_rel>=0.7, skills>=3, 4<=yoe<=10, not honeypot)")
print(f"  Found: {len(low_entail_high_quality)}")

# How many have weak labels?
with_wl = [(c, e, t, s, y, w) for c, e, t, s, y, w in low_entail_high_quality if w is not None]
print(f"  With weak labels: {len(with_wl)}")
if with_wl:
    wl_scores_entail = [w for _, _, _, _, _, w in with_wl]
    print(f"  Their avg weak label: {np.mean(wl_scores_entail):.2f}")
    print(f"  Their weak label distribution: {Counter(int(w) for _, _, _, _, _, w in with_wl)}")
    print(f"\n  Details:")
    for cid, entail, title_rel, skills, yoe, wl_s in sorted(with_wl, key=lambda x: -(x[5] or 0))[:15]:
        in_sub = "IN SUB" if cid in set(submission_cids) else "NOT IN"
        print(f"    {cid} entail={entail:.2f} title_rel={title_rel:.2f} skills={skills:.0f} yoe={yoe:.1f} wl={wl_s:.0f} {in_sub}")

# ============================================================================
# 8. CUSTOMER SUPPORT ANALYSIS
# ============================================================================
print(f"\n{SEP}")
print("8. CUSTOMER SUPPORT CANDIDATES IN SUBMISSION")
print(SEP)

# Load actual candidates for the CS titles
candidates_by_id = {}
with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        c = json.loads(line)
        if c["candidate_id"] in ["CAND_0057340", "CAND_0026316"]:
            candidates_by_id[c["candidate_id"]] = c

for cid in ["CAND_0057340", "CAND_0026316"]:
    if cid in candidates_by_id:
        c = candidates_by_id[cid]
        p = c.get("profile", {})
        skills = c.get("skills", [])
        career = c.get("career_history", [])
        signals = c.get("redrob_signals", {})
        
        print(f"\n{cid}:")
        print(f"  Title: {p.get('current_title')}")
        print(f"  Company: {p.get('current_company')}")
        print(f"  YoE: {p.get('years_of_experience')}")
        print(f"  Location: {p.get('location')}, {p.get('country')}")
        print(f"  Headline: {p.get('headline', 'N/A')[:100]}")
        
        skill_names = [s.get("name") for s in skills[:10]]
        print(f"  Top skills: {', '.join(skill_names)}")
        
        for job in career[:3]:
            print(f"  Career: {job.get('title')} at {job.get('company')} ({job.get('duration_months')}mo)")
            desc = (job.get("description") or "")[:150]
            print(f"    Desc: {desc}")
        
        print(f"  Response rate: {signals.get('recruiter_response_rate')}")
        print(f"  Notice period: {signals.get('notice_period_days')}d")
        print(f"  Saved by recruiters: {signals.get('saved_by_recruiters_30d')}")
        
        # Check weak label
        if cid in wl_lookup:
            print(f"  WEAK LABEL: {wl_lookup[cid]}")
        
        # Check honeypot
        feats = feat_lookup.get(cid, {})
        print(f"  Honeypot flag: {feats.get('is_honeypot', 'N/A')}")
        print(f"  Honeypot flag count: {feats.get('honeypot_flag_count', 'N/A')}")
        print(f"  Entailment: {feats.get('skill_text_entailment_rate', 'N/A')}")

print(f"\n{SEP}")
print("ANALYSIS COMPLETE")
print(SEP)
