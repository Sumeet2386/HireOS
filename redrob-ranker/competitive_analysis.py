"""Deep competitive analysis: Can this submission win?
Analyzes the submission against multiple ground-truth hypotheses."""

import csv, json, math, numpy as np
from pathlib import Path
from collections import Counter
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
CANDS_PATH = ROOT.parent / "India_runs_data_and_ai_challenge" / "candidates.jsonl"

# Load submission
def load_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row["candidate_id"].strip())
    return rows

sub = load_csv(ROOT / "final_submission.csv")

# Load weak labels
with open(ARTIFACTS / "weak_labels.json") as f:
    wl = json.load(f)
wl_lookup = {w["candidate_id"]: w["weak_label"] for w in wl}

# Load features
feat_table = pq.read_table(ARTIFACTS / "features.parquet")
feat_dict = feat_table.to_pydict()
feat_cids = feat_dict.pop("candidate_id")
feat_cols = list(feat_dict.keys())
feat_lookup = {}
for i, cid in enumerate(feat_cids):
    feat_lookup[cid] = {col: feat_dict[col][i] for col in feat_cols}

# Load candidates (only those in submission or high-quality)
candidates = {}
with open(CANDS_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip(): continue
        c = json.loads(line)
        candidates[c["candidate_id"]] = c

SEP = "=" * 70

# ============================================================================
# 1. QUALITY SCORE FOR EACH CANDIDATE
# ============================================================================
print(SEP)
print("1. QUALITY ANALYSIS OF TOP 100")
print(SEP)

def compute_quality_score(cid):
    """Compute a composite quality score mimicking what an expert judge would rate."""
    feats = feat_lookup.get(cid, {})
    cand = candidates.get(cid, {})
    p = cand.get("profile", {})
    s = cand.get("redrob_signals", {})
    
    # Title relevance (0-1)
    title_r = feats.get("current_title_relevance", 0)
    
    # Skill depth
    core_ai = feats.get("num_core_ai_skills", 0)
    high_signal = feats.get("num_high_signal_skills", 0)
    proficiency = feats.get("skill_proficiency_score", 0)
    entailment = feats.get("skill_text_entailment_rate", 0)
    
    # Experience fit
    yoe = feats.get("years_of_experience", 0)
    yoe_fit = feats.get("yoe_in_ideal_range", 0)
    
    # Company quality
    product_co = feats.get("product_company_count", 0)
    has_product = feats.get("has_product_company_exp", 0)
    
    # Behavioral
    saved = s.get("saved_by_recruiters_30d", 0) if s else 0
    search = s.get("search_appearance_30d", 0) if s else 0
    response = s.get("recruiter_response_rate", 0) if s else 0
    
    # Location
    is_india = feats.get("is_india", 0)
    
    # Honeypot risk
    is_hp = feats.get("is_honeypot", 0)
    hp_flags = feats.get("honeypot_flag_count", 0)
    
    # Composite quality (expert-like scoring)
    quality = (
        title_r * 25 +  # Title is the strongest signal
        min(core_ai / 8, 1.0) * 20 +  # Core AI skill breadth
        proficiency * 10 +  # Skill depth
        entailment * 8 +  # Skill authenticity
        yoe_fit * 10 +  # Experience fit
        min(product_co / 2, 1.0) * 8 +  # Product company background
        min(saved / 30, 1.0) * 7 +  # Recruiter validation
        min(search / 300, 1.0) * 5 +  # Market demand
        response * 4 +  # Responsiveness
        is_india * 3  # Location
    )
    
    # Penalties
    if is_hp > 0:
        quality *= 0.1  # Near-zero for honeypots
    if hp_flags >= 5:
        quality *= 0.5
    
    return quality

# Score all candidates
all_quality = []
for cid in feat_lookup:
    q = compute_quality_score(cid)
    all_quality.append((cid, q))
all_quality.sort(key=lambda x: -x[1])

# Score submission candidates
sub_quality = [(cid, compute_quality_score(cid)) for cid in sub]
sub_quality_scores = [q for _, q in sub_quality]

print(f"Submission quality scores:")
print(f"  Top 10 avg: {np.mean(sub_quality_scores[:10]):.1f}")
print(f"  Top 50 avg: {np.mean(sub_quality_scores[:50]):.1f}")
print(f"  Full 100 avg: {np.mean(sub_quality_scores):.1f}")
print(f"  Min: {min(sub_quality_scores):.1f}, Max: {max(sub_quality_scores):.1f}")

# What's the best possible top 100?
ideal_100 = all_quality[:100]
ideal_quality_scores = [q for _, q in ideal_100]

print(f"\nIdeal top 100 (by quality score):")
print(f"  Top 10 avg: {np.mean(ideal_quality_scores[:10]):.1f}")
print(f"  Top 50 avg: {np.mean(ideal_quality_scores[:50]):.1f}")
print(f"  Full 100 avg: {np.mean(ideal_quality_scores):.1f}")

# Overlap
sub_set = set(sub)
ideal_set = set(cid for cid, _ in ideal_100)
overlap = sub_set & ideal_set
print(f"\nOverlap with ideal: {len(overlap)}/100 ({len(overlap)}%)")

# Top 10 overlap
ideal_top10 = set(cid for cid, _ in ideal_100[:10])
sub_top10 = set(sub[:10])
print(f"Top 10 overlap with ideal: {len(sub_top10 & ideal_top10)}/10")

# ============================================================================
# 2. SIMULATED NDCG
# ============================================================================
print(f"\n{SEP}")
print("2. SIMULATED NDCG (using quality scores as ground truth proxy)")
print(SEP)

# Build ground truth relevance: all candidates, quality as relevance
gt_lookup = {cid: q for cid, q in all_quality}

def dcg_at_k(relevances, k):
    return sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(relevances[:k]))

def ndcg_at_k(our_rels, all_rels_sorted, k):
    actual = dcg_at_k(our_rels, k)
    ideal = dcg_at_k(all_rels_sorted[:k], k)
    return actual / ideal if ideal > 0 else 0.0

# Normalize quality scores to 0-10 range for NDCG
max_q = max(q for _, q in all_quality)
our_rels = [gt_lookup.get(cid, 0) / max_q * 10 for cid in sub]
all_sorted = sorted([q / max_q * 10 for _, q in all_quality], reverse=True)

ndcg10 = ndcg_at_k(our_rels, all_sorted, 10)
ndcg50 = ndcg_at_k(our_rels, all_sorted, 50)
map_approx = ndcg_at_k(our_rels, all_sorted, 100)  # Simplified MAP proxy

# Composite score (the actual eval formula)
p_at_10 = sum(1 for r in our_rels[:10] if r >= 7) / 10
composite = 0.50 * ndcg10 + 0.30 * ndcg50 + 0.15 * map_approx + 0.05 * p_at_10

print(f"NDCG@10:  {ndcg10:.4f}  (50% weight)")
print(f"NDCG@50:  {ndcg50:.4f}  (30% weight)")
print(f"NDCG@100: {map_approx:.4f}  (proxy for MAP, 15% weight)")
print(f"P@10:     {p_at_10:.4f}  (5% weight)")
print(f"\nEstimated composite: {composite:.4f}")

# ============================================================================
# 3. RANK-BY-RANK QUALITY ANALYSIS
# ============================================================================
print(f"\n{SEP}")
print("3. TOP 20 DETAILED QUALITY BREAKDOWN")
print(SEP)

print(f"{'Rank':>4} {'ID':>15} {'Quality':>8} {'Title':>35} {'YoE':>5} {'Saved':>6} {'Search':>7} {'Entail':>7} {'HP':>3}")
for i in range(20):
    cid = sub[i]
    q = compute_quality_score(cid)
    c = candidates.get(cid, {})
    p = c.get("profile", {})
    s = c.get("redrob_signals", {})
    f = feat_lookup.get(cid, {})
    title = (p.get("current_title", "?"))[:35]
    yoe = p.get("years_of_experience", 0)
    saved = s.get("saved_by_recruiters_30d", 0) if s else 0
    search = s.get("search_appearance_30d", 0) if s else 0
    entail = f.get("skill_text_entailment_rate", 0)
    hp = int(f.get("is_honeypot", 0))
    print(f"  #{i+1:>2} {cid} {q:>8.1f} {title:>35} {yoe:>5.1f} {saved:>6} {search:>7} {entail:>7.2f} {hp:>3}")

# ============================================================================
# 4. WHAT'S MISSING? - CANDIDATES WE SHOULD HAVE
# ============================================================================
print(f"\n{SEP}")
print("4. BEST CANDIDATES NOT IN SUBMISSION (potential misses)")
print(SEP)

missed = [(cid, q) for cid, q in all_quality if cid not in sub_set][:20]
print(f"{'Rank':>4} {'ID':>15} {'Quality':>8} {'Title':>35} {'YoE':>5} {'Saved':>6} {'HP':>3} {'HPFlags':>7}")
for i, (cid, q) in enumerate(missed):
    c = candidates.get(cid, {})
    p = c.get("profile", {})
    s = c.get("redrob_signals", {})
    f = feat_lookup.get(cid, {})
    title = (p.get("current_title", "?"))[:35]
    yoe = p.get("years_of_experience", 0)
    saved = s.get("saved_by_recruiters_30d", 0) if s else 0
    hp = int(f.get("is_honeypot", 0))
    hp_flags = int(f.get("honeypot_flag_count", 0))
    print(f"  #{i+1:>2} {cid} {q:>8.1f} {title:>35} {yoe:>5.1f} {saved:>6} {hp:>3} {hp_flags:>7}")

# ============================================================================
# 5. WORST CANDIDATES IN SUBMISSION (potential to swap)
# ============================================================================
print(f"\n{SEP}")
print("5. WEAKEST CANDIDATES IN SUBMISSION (bottom 20)")
print(SEP)

sub_sorted = sorted(enumerate(sub), key=lambda x: compute_quality_score(x[1]))
print(f"{'SubRank':>7} {'ID':>15} {'Quality':>8} {'Title':>35} {'YoE':>5} {'HP':>3}")
for i, (rank_idx, cid) in enumerate(sub_sorted[:20]):
    q = compute_quality_score(cid)
    c = candidates.get(cid, {})
    p = c.get("profile", {})
    f = feat_lookup.get(cid, {})
    title = (p.get("current_title", "?"))[:35]
    yoe = p.get("years_of_experience", 0)
    hp = int(f.get("is_honeypot", 0))
    print(f"  #{rank_idx+1:>4} {cid} {q:>8.1f} {title:>35} {yoe:>5.1f} {hp:>3}")

# ============================================================================
# 6. COMPETITIVE ANALYSIS - HOW GOOD IS GOOD ENOUGH?
# ============================================================================
print(f"\n{SEP}")
print("6. COMPETITIVE ANALYSIS")
print(SEP)

# What's the theoretical max composite score?
perfect_rels = all_sorted[:100]
perfect_ndcg10 = ndcg_at_k(perfect_rels, all_sorted, 10)
perfect_ndcg50 = ndcg_at_k(perfect_rels, all_sorted, 50)
perfect_map = ndcg_at_k(perfect_rels, all_sorted, 100)
perfect_p10 = sum(1 for r in perfect_rels[:10] if r >= 7) / 10
perfect_composite = 0.50 * perfect_ndcg10 + 0.30 * perfect_ndcg50 + 0.15 * perfect_map + 0.05 * perfect_p10

print(f"Perfect score (ideal ranking): {perfect_composite:.4f}")
print(f"Our score:                     {composite:.4f}")
print(f"Our efficiency:                {composite/perfect_composite*100:.1f}%")

# Random baseline
rng = np.random.default_rng(42)
random_scores = []
all_cids = list(feat_lookup.keys())
for _ in range(100):
    random_sub = rng.choice(all_cids, 100, replace=False)
    random_rels = [gt_lookup.get(cid, 0) / max_q * 10 for cid in random_sub]
    r_ndcg10 = ndcg_at_k(random_rels, all_sorted, 10)
    r_ndcg50 = ndcg_at_k(random_rels, all_sorted, 50)
    r_map = ndcg_at_k(random_rels, all_sorted, 100)
    r_p10 = sum(1 for r in random_rels[:10] if r >= 7) / 10
    random_scores.append(0.50 * r_ndcg10 + 0.30 * r_ndcg50 + 0.15 * r_map + 0.05 * r_p10)

print(f"\nRandom baseline (100 trials):")
print(f"  Mean: {np.mean(random_scores):.4f}")
print(f"  Best: {max(random_scores):.4f}")
print(f"  Our advantage over random: {(composite - np.mean(random_scores)) / np.mean(random_scores) * 100:.1f}%")

# Naive sort by saved_by_recruiters (the LightGBM strategy)
naive_saved = sorted(all_cids, key=lambda cid: -feat_lookup[cid].get("saved_by_recruiters_30d", 0))[:100]
naive_rels = [gt_lookup.get(cid, 0) / max_q * 10 for cid in naive_saved]
naive_ndcg10 = ndcg_at_k(naive_rels, all_sorted, 10)
naive_ndcg50 = ndcg_at_k(naive_rels, all_sorted, 50)
naive_map = ndcg_at_k(naive_rels, all_sorted, 100)
naive_p10 = sum(1 for r in naive_rels[:10] if r >= 7) / 10
naive_composite = 0.50 * naive_ndcg10 + 0.30 * naive_ndcg50 + 0.15 * naive_map + 0.05 * naive_p10

print(f"\nNaive (sort by saved_by_recruiters):")
print(f"  Composite: {naive_composite:.4f}")
print(f"  vs Our: {'BETTER' if composite > naive_composite else 'WORSE'} by {abs(composite - naive_composite):.4f}")

# Naive sort by title_relevance + skills
naive_quality = sorted(all_cids, key=lambda cid: (
    -feat_lookup[cid].get("current_title_relevance", 0),
    -feat_lookup[cid].get("num_core_ai_skills", 0),
    -feat_lookup[cid].get("skill_proficiency_score", 0),
))[:100]
naive_q_rels = [gt_lookup.get(cid, 0) / max_q * 10 for cid in naive_quality]
naive_q_ndcg10 = ndcg_at_k(naive_q_rels, all_sorted, 10)
naive_q_ndcg50 = ndcg_at_k(naive_q_rels, all_sorted, 50)
naive_q_map = ndcg_at_k(naive_q_rels, all_sorted, 100)
naive_q_p10 = sum(1 for r in naive_q_rels[:10] if r >= 7) / 10
naive_q_composite = 0.50 * naive_q_ndcg10 + 0.30 * naive_q_ndcg50 + 0.15 * naive_q_map + 0.05 * naive_q_p10

print(f"\nNaive (sort by title+skills+proficiency):")
print(f"  Composite: {naive_q_composite:.4f}")
print(f"  vs Our: {'BETTER' if composite > naive_q_composite else 'WORSE'} by {abs(composite - naive_q_composite):.4f}")

# What if we just sorted by our quality score?
oracle_sub = [cid for cid, _ in all_quality[:100]]
oracle_rels = [gt_lookup.get(cid, 0) / max_q * 10 for cid in oracle_sub]
oracle_composite = 0.50 * ndcg_at_k(oracle_rels, all_sorted, 10) + 0.30 * ndcg_at_k(oracle_rels, all_sorted, 50) + 0.15 * ndcg_at_k(oracle_rels, all_sorted, 100) + 0.05 * (sum(1 for r in oracle_rels[:10] if r >= 7) / 10)
print(f"\nOracle (sort by our quality composite):")
print(f"  Composite: {oracle_composite:.4f}")

print(f"\n{SEP}")
print("SUMMARY")
print(SEP)
print(f"  Our composite:         {composite:.4f}")
print(f"  Perfect:               {perfect_composite:.4f}")
print(f"  Efficiency:            {composite/perfect_composite*100:.1f}%")
print(f"  vs Random:             +{(composite - np.mean(random_scores))/np.mean(random_scores)*100:.0f}%")
print(f"  vs Naive (saved):      {'WIN' if composite > naive_composite else 'LOSE'} ({composite:.4f} vs {naive_composite:.4f})")
print(f"  vs Naive (title+skill): {'WIN' if composite > naive_q_composite else 'LOSE'} ({composite:.4f} vs {naive_q_composite:.4f})")
print(f"\n  Top 100 overlap with ideal: {len(overlap)}/100")
print(f"  Top 10 overlap with ideal:  {len(sub_top10 & ideal_top10)}/10")
