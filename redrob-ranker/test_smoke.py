"""Quick smoke test on sample candidates."""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ranker.features import extract_all_features
from ranker.honeypot import detect_honeypot
from ranker.reasoning import build_template_reasoning

SAMPLE_PATH = Path(__file__).resolve().parent.parent / "India_runs_data_and_ai_challenge" / "sample_candidates.json"

data = json.load(open(SAMPLE_PATH, encoding="utf-8"))

print(f"Testing on {len(data)} sample candidates...\n")

for i, cand in enumerate(data[:5]):
    cid = cand["candidate_id"]
    title = cand["profile"]["current_title"]
    yoe = cand["profile"]["years_of_experience"]

    features = extract_all_features(cand)
    is_hp, hp_flags = detect_honeypot(cand)
    reasoning = build_template_reasoning(cand, rank=i+1, score=0.9)

    print(f"--- {cid} ({title}, {yoe}y) ---")
    print(f"  Honeypot: {is_hp} ({len(hp_flags)} flags)")
    if hp_flags:
        for f in hp_flags[:3]:
            print(f"    - {f}")
    print(f"  Title relevance: {features['current_title_relevance']:.2f}")
    print(f"  YoE ideal range: {features['yoe_in_ideal_range']:.2f}")
    print(f"  Core AI skills: {features['num_core_ai_skills']:.0f}")
    print(f"  Location fit: {features['location_fit']:.2f}")
    print(f"  Activity decay: {features['activity_decay_score']:.3f}")
    print(f"  Product company exp: {features['has_product_company_exp']:.0f}")
    print(f"  All consulting: {features['all_consulting_career']:.0f}")
    print(f"  Reasoning: {reasoning[:120]}...")
    print()

print("All tests passed!")
