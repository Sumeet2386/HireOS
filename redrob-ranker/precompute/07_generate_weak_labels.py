"""
07_generate_weak_labels.py — Generate weak supervision labels using GPT-4o-mini.

Scores a stratified sample of ~2,000 candidates on a 0-10 relevance scale
using the JD context. These labels train the LightGBM LTR model.

Usage:
    python precompute/07_generate_weak_labels.py --features <parquet_path> --candidates <jsonl> --out <output_dir> [--n 2000]

Requires OPENAI_API_KEY environment variable.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

# Add parent to path for ranker imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


JD_CONTEXT = """
You are evaluating candidates for a Senior AI Engineer role at a startup building
an AI-powered talent intelligence platform (search, ranking, recommendations).

Key requirements:
- 5-9 years of experience in AI/ML engineering
- Production experience with embeddings, semantic search, retrieval systems
- Hands-on with vector databases (FAISS, Pinecone, Weaviate)
- Strong NLP/transformers background (BERT, GPT, fine-tuning)
- Ranking/recommendation systems experience (LTR, NDCG, A/B testing)
- Python, PyTorch or TensorFlow, MLOps
- India-based preferred (Pune, Noida, Hyderabad, Mumbai, Delhi NCR)
- Product company background preferred over pure consulting
- Active on platform, responsive, available (short notice period)

Red flags (honeypots):
- Non-technical titles with AI keywords stuffed in skills
- Career timeline impossibilities
- Skills claimed but not evidenced in career descriptions
"""

SCORING_PROMPT = """
Rate this candidate's fit for the Senior AI Engineer role on a scale of 0-10.

SCORING GUIDE:
- 0-2: Completely irrelevant (non-technical, no AI skills, wrong domain)
- 3-4: Marginally relevant (some tech skills but wrong specialization)
- 5-6: Moderately relevant (adjacent technical role, some AI exposure)
- 7-8: Strong fit (relevant title, solid AI/ML skills, good experience)
- 9-10: Exceptional fit (perfect title, deep AI/search/ranking expertise, product company)

CANDIDATE PROFILE:
{profile_text}

Respond with ONLY a JSON object: {{"score": <0-10>, "reason": "<one sentence>"}}
"""


def build_profile_text(candidate: dict) -> str:
    """Build a concise profile summary for the LLM."""
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})

    parts = [
        f"Title: {profile.get('current_title', '?')}",
        f"YoE: {profile.get('years_of_experience', 0):.1f}",
        f"Company: {profile.get('current_company', '?')} ({profile.get('current_industry', '?')})",
        f"Location: {profile.get('location', '?')}, {profile.get('country', '?')}",
        f"Headline: {profile.get('headline', 'N/A')}",
    ]

    # Top skills
    skill_names = [s["name"] for s in skills[:15]]
    parts.append(f"Skills: {', '.join(skill_names)}")

    # Career history (abbreviated)
    for job in career[:3]:
        desc = (job.get("description") or "")[:200]
        parts.append(f"  - {job.get('title', '?')} at {job.get('company', '?')} "
                     f"({job.get('duration_months', 0)}mo): {desc}")

    # Behavioral signals
    parts.append(f"Response rate: {signals.get('recruiter_response_rate', 0):.2f}")
    parts.append(f"Notice period: {signals.get('notice_period_days', 0)}d")
    parts.append(f"Open to work: {signals.get('open_to_work_flag', False)}")

    return "\n".join(parts)


def score_candidates_batch(candidates: list[dict], api_key: str) -> list[dict]:
    """Score a batch of candidates using OpenAI API."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    results = []

    for i, cand in enumerate(candidates):
        cid = cand["candidate_id"]
        profile_text = build_profile_text(cand)
        prompt = SCORING_PROMPT.format(profile_text=profile_text)

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": JD_CONTEXT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=100,
                response_format={"type": "json_object"},
            )

            resp_text = response.choices[0].message.content.strip()
            resp_json = json.loads(resp_text)
            score = float(resp_json.get("score", 5))
            reason = resp_json.get("reason", "")

            results.append({
                "candidate_id": cid,
                "weak_label": score,
                "reason": reason,
            })

        except Exception as e:
            print(f"  ERROR scoring {cid}: {e}")
            results.append({
                "candidate_id": cid,
                "weak_label": 5.0,  # Neutral fallback
                "reason": f"Error: {str(e)[:50]}",
            })

        # Rate limiting
        if (i + 1) % 50 == 0:
            print(f"  Scored {i+1}/{len(candidates)}...")
            time.sleep(1)

    return results


def select_stratified_sample(
    features_path: Path,
    candidates_path: Path,
    n: int = 2000,
) -> list[dict]:
    """
    Select a stratified sample of candidates for weak labeling.

    Stratification ensures we get candidates from all quality tiers,
    not just the middle of the distribution.
    """
    # Load features for stratification
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(features_path)
        df = table.to_pydict()
        cids = df["candidate_id"]
        title_relevance = df.get("current_title_relevance", [0.0] * len(cids))
        yoe_fit = df.get("yoe_in_ideal_range", [0.0] * len(cids))
        ai_skills = df.get("num_core_ai_skills", [0.0] * len(cids))
    except Exception:
        # Fallback: random sampling
        print("  WARNING: Cannot load features for stratification. Using random sampling.")
        candidates = []
        with open(candidates_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    candidates.append(json.loads(line))
        random.shuffle(candidates)
        return candidates[:n]

    # Build quality proxy score for stratification
    quality_scores = []
    for i in range(len(cids)):
        q = (title_relevance[i] * 0.4 + yoe_fit[i] * 0.3 + min(ai_skills[i] / 10, 1.0) * 0.3)
        quality_scores.append((cids[i], q))

    # Sort and stratify into 5 tiers
    quality_scores.sort(key=lambda x: x[1])
    tier_size = len(quality_scores) // 5
    per_tier = n // 5

    selected_ids = set()
    for tier in range(5):
        start = tier * tier_size
        end = start + tier_size if tier < 4 else len(quality_scores)
        tier_cids = [cid for cid, _ in quality_scores[start:end]]
        sampled = random.sample(tier_cids, min(per_tier, len(tier_cids)))
        selected_ids.update(sampled)

    # Fill up to n if needed
    remaining = [cid for cid, _ in quality_scores if cid not in selected_ids]
    random.shuffle(remaining)
    while len(selected_ids) < n and remaining:
        selected_ids.add(remaining.pop())

    # Load selected candidates from JSONL
    candidates = []
    with open(candidates_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            cand = json.loads(line)
            if cand["candidate_id"] in selected_ids:
                candidates.append(cand)

    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate weak labels via GPT-4o-mini.")
    parser.add_argument("--features", default="artifacts/features.parquet",
                        help="Path to features.parquet for stratified sampling")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--out", default="artifacts", help="Output directory")
    parser.add_argument("--n", type=int, default=2000, help="Number of candidates to label")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        print("Set it with: export OPENAI_API_KEY=sk-...")
        sys.exit(1)

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Selecting {args.n} candidates for weak labeling...")
    candidates = select_stratified_sample(
        Path(args.features), Path(args.candidates), args.n
    )
    print(f"Selected {len(candidates)} candidates.")

    print(f"\nScoring via GPT-4o-mini...")
    results = score_candidates_batch(candidates, api_key)

    # Save results
    output_path = output_dir / "weak_labels.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} weak labels to {output_path}")

    # Print distribution
    scores = [r["weak_label"] for r in results]
    print(f"\nScore distribution:")
    for bucket in range(0, 11, 2):
        count = sum(1 for s in scores if bucket <= s < bucket + 2)
        print(f"  {bucket}-{bucket+1}: {count}")

    avg = sum(scores) / len(scores)
    print(f"\nAverage score: {avg:.2f}")


if __name__ == "__main__":
    main()
