"""
02_text_synthesis.py — Concatenate profile text per candidate.

Creates a unified text document per candidate for embedding generation.
Also creates a separate career-only text for BM25 indexing.

Usage:
    python precompute/02_text_synthesis.py --candidates <path> --out <output_dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Try to use pyarrow for Parquet; fall back to CSV
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False
    print("WARNING: pyarrow not installed. Will output CSV instead of Parquet.")


def synthesize_text(candidate: dict) -> tuple[str, str]:
    """
    Create unified text documents from a candidate record.

    Returns
    -------
    tuple[str, str]
        (full_text, career_text)
        full_text: Complete profile for embedding generation
        career_text: Career descriptions only for BM25
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    education = candidate.get("education", [])

    parts = []

    # Headline and summary (most semantically rich)
    headline = (profile.get("headline") or "").strip()
    summary = (profile.get("summary") or "").strip()
    if headline:
        parts.append(headline)
    if summary:
        parts.append(summary)

    # Current title and company
    current_title = (profile.get("current_title") or "").strip()
    current_company = (profile.get("current_company") or "").strip()
    if current_title:
        parts.append(f"Currently working as {current_title} at {current_company}.")

    # Career history
    career_parts = []
    for job in career:
        company = (job.get("company") or "").strip()
        title = (job.get("title") or "").strip()
        duration = job.get("duration_months", 0)
        description = (job.get("description") or "").strip()
        industry = (job.get("industry") or "").strip()

        job_text = f"At {company} as {title}"
        if duration:
            job_text += f" ({duration} months)"
        if industry:
            job_text += f" in {industry}"
        job_text += f": {description}" if description else "."

        career_parts.append(job_text)

    if career_parts:
        parts.extend(career_parts)

    # Skills
    skill_names = [s.get("name", "") for s in skills if s.get("name")]
    if skill_names:
        parts.append(f"Skills: {', '.join(skill_names)}.")

    # Education
    for edu in education:
        degree = (edu.get("degree") or "").strip()
        field = (edu.get("field_of_study") or "").strip()
        institution = (edu.get("institution") or "").strip()
        if degree and field:
            parts.append(f"{degree} in {field} from {institution}.")

    full_text = " ".join(parts)
    career_text = " ".join(career_parts)

    # Normalize whitespace
    full_text = " ".join(full_text.split())
    career_text = " ".join(career_text.split())

    return full_text, career_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize text per candidate.")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--out", default="artifacts", help="Output directory")
    args = parser.parse_args()

    input_path = Path(args.candidates)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_ids: list[str] = []
    full_texts: list[str] = []
    career_texts: list[str] = []

    total = 0
    print(f"Reading {input_path}...")

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            candidate = json.loads(line)

            cid = candidate.get("candidate_id", "")
            full_text, career_text = synthesize_text(candidate)

            candidate_ids.append(cid)
            full_texts.append(full_text)
            career_texts.append(career_text)

            if total % 10000 == 0:
                print(f"  Processed {total:,} candidates...")

    print(f"Total: {total:,} candidates synthesized.")

    if HAS_PARQUET:
        table = pa.table({
            "candidate_id": candidate_ids,
            "full_text": full_texts,
            "career_text": career_texts,
        })
        output_path = output_dir / "candidate_texts.parquet"
        pq.write_table(table, output_path)
        print(f"Saved to {output_path}")
    else:
        import csv
        output_path = output_dir / "candidate_texts.csv"
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["candidate_id", "full_text", "career_text"])
            for cid, ft, ct in zip(candidate_ids, full_texts, career_texts):
                writer.writerow([cid, ft, ct])
        print(f"Saved to {output_path}")

    # Also save the ID mapping (row index → candidate_id)
    mapping = {i: cid for i, cid in enumerate(candidate_ids)}
    mapping_path = output_dir / "id_mapping.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f)
    print(f"ID mapping saved to {mapping_path}")


if __name__ == "__main__":
    main()
