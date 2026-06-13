"""
01_eda.py — Exploratory Data Analysis on the full 100K dataset.

Outputs statistical summaries needed to calibrate decay functions and
normalize features in the online phase.

Usage:
    python precompute/01_eda.py --candidates <path> --out <output_dir>
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path


def quantile(values: list[float], pct: float) -> float:
    """Simple percentile calculation."""
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * pct))))
    return ordered[idx]


def main() -> None:
    parser = argparse.ArgumentParser(description="EDA on candidate dataset.")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--out", default="artifacts", help="Output directory for stats")
    args = parser.parse_args()

    input_path = Path(args.candidates)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collectors
    exp_vals: list[float] = []
    response_rate_vals: list[float] = []
    response_time_vals: list[float] = []
    notice_vals: list[int] = []
    github_vals: list[float] = []
    saved_vals: list[int] = []
    views_vals: list[int] = []
    search_vals: list[int] = []
    completeness_vals: list[float] = []
    interview_vals: list[float] = []
    connection_vals: list[int] = []
    endorsement_vals: list[int] = []
    skill_count_vals: list[int] = []

    title_counter: Counter[str] = Counter()
    skill_counter: Counter[str] = Counter()
    country_counter: Counter[str] = Counter()
    industry_counter: Counter[str] = Counter()
    proficiency_counter: Counter[str] = Counter()
    company_size_counter: Counter[str] = Counter()
    work_mode_counter: Counter[str] = Counter()

    total = 0
    has_assessment = 0
    open_to_work_count = 0
    willing_to_relocate_count = 0

    print(f"Reading {input_path}...")

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            candidate = json.loads(line)

            profile = candidate.get("profile", {})
            signals = candidate.get("redrob_signals", {})
            skills = candidate.get("skills", [])
            career = candidate.get("career_history", [])

            # Profile stats
            exp_vals.append(profile.get("years_of_experience", 0))
            title_counter[profile.get("current_title", "unknown")] += 1
            country_counter[profile.get("country", "unknown")] += 1
            industry_counter[profile.get("current_industry", "unknown")] += 1
            company_size_counter[profile.get("current_company_size", "unknown")] += 1

            # Skill stats
            skill_count_vals.append(len(skills))
            for skill in skills:
                skill_counter[skill.get("name", "unknown")] += 1
                proficiency_counter[skill.get("proficiency", "unknown")] += 1

            # Behavioral stats
            response_rate_vals.append(signals.get("recruiter_response_rate", 0))
            response_time_vals.append(signals.get("avg_response_time_hours", 0))
            notice_vals.append(signals.get("notice_period_days", 0))
            github_score = signals.get("github_activity_score", -1)
            if github_score >= 0:
                github_vals.append(github_score)
            saved_vals.append(signals.get("saved_by_recruiters_30d", 0))
            views_vals.append(signals.get("profile_views_received_30d", 0))
            search_vals.append(signals.get("search_appearance_30d", 0))
            completeness_vals.append(signals.get("profile_completeness_score", 0))
            interview_vals.append(signals.get("interview_completion_rate", 0))
            connection_vals.append(signals.get("connection_count", 0))
            endorsement_vals.append(signals.get("endorsements_received", 0))
            work_mode_counter[signals.get("preferred_work_mode", "unknown")] += 1

            if signals.get("skill_assessment_scores"):
                has_assessment += 1
            if signals.get("open_to_work_flag"):
                open_to_work_count += 1
            if signals.get("willing_to_relocate"):
                willing_to_relocate_count += 1

            if total % 10000 == 0:
                print(f"  Processed {total:,} candidates...")

    print(f"\nTotal candidates: {total:,}")

    # Compute stats
    stats = {
        "total_candidates": total,
        "experience": {
            "mean": round(statistics.mean(exp_vals), 2),
            "std": round(statistics.stdev(exp_vals), 2) if len(exp_vals) > 1 else 0,
            "p10": round(quantile(exp_vals, 0.10), 1),
            "p25": round(quantile(exp_vals, 0.25), 1),
            "p50": round(quantile(exp_vals, 0.50), 1),
            "p75": round(quantile(exp_vals, 0.75), 1),
            "p90": round(quantile(exp_vals, 0.90), 1),
        },
        "response_rate": {
            "mean": round(statistics.mean(response_rate_vals), 3),
            "p25": round(quantile(response_rate_vals, 0.25), 3),
            "p50": round(quantile(response_rate_vals, 0.50), 3),
            "p75": round(quantile(response_rate_vals, 0.75), 3),
        },
        "response_time_hours": {
            "mean": round(statistics.mean(response_time_vals), 1),
            "p50": round(quantile(response_time_vals, 0.50), 1),
        },
        "notice_period_days": {
            "p25": quantile(notice_vals, 0.25),
            "p50": quantile(notice_vals, 0.50),
            "p75": quantile(notice_vals, 0.75),
        },
        "github_activity": {
            "count_with_github": len(github_vals),
            "mean": round(statistics.mean(github_vals), 1) if github_vals else 0,
        },
        "saved_by_recruiters": {
            "p50": quantile(saved_vals, 0.50),
            "p90": quantile(saved_vals, 0.90),
        },
        "profile_completeness": {
            "mean": round(statistics.mean(completeness_vals), 1),
        },
        "skill_count": {
            "mean": round(statistics.mean(skill_count_vals), 1),
            "p50": quantile(skill_count_vals, 0.50),
        },
        "flags": {
            "has_assessment_count": has_assessment,
            "open_to_work_count": open_to_work_count,
            "willing_to_relocate_count": willing_to_relocate_count,
        },
        "top_titles": dict(title_counter.most_common(20)),
        "top_skills": dict(skill_counter.most_common(30)),
        "top_countries": dict(country_counter.most_common(10)),
        "top_industries": dict(industry_counter.most_common(15)),
        "company_sizes": dict(company_size_counter),
        "work_modes": dict(work_mode_counter),
        "proficiency_distribution": dict(proficiency_counter),
    }

    # Save JSON
    stats_path = output_dir / "eda_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nEDA stats saved to {stats_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"DATASET PROFILE SUMMARY")
    print(f"{'='*60}")
    print(f"Total candidates: {total:,}")
    print(f"Experience: mean={stats['experience']['mean']}, "
          f"p10={stats['experience']['p10']}, p50={stats['experience']['p50']}, "
          f"p90={stats['experience']['p90']}")
    print(f"Response rate: mean={stats['response_rate']['mean']}")
    print(f"Notice period: p25={stats['notice_period_days']['p25']}, "
          f"p50={stats['notice_period_days']['p50']}, "
          f"p75={stats['notice_period_days']['p75']}")
    print(f"Has assessments: {has_assessment:,} ({has_assessment/total*100:.1f}%)")
    print(f"Open to work: {open_to_work_count:,} ({open_to_work_count/total*100:.1f}%)")
    print(f"\nTop 10 titles:")
    for title, count in title_counter.most_common(10):
        print(f"  {title}: {count:,}")
    print(f"\nTop 10 countries:")
    for country, count in country_counter.most_common(10):
        print(f"  {country}: {count:,}")


if __name__ == "__main__":
    main()
