"""
Redrob AI Candidate Ranking System — HuggingFace Spaces Demo

Accepts a small candidate sample (≤100 candidates) as JSON or JSONL,
runs the full ranking pipeline, and produces a downloadable ranked CSV.

This is the sandbox/demo environment for the hackathon submission.
For the full 100K pipeline with FAISS/BM25 recall, see the main repo.
"""

import csv
import io
import json
import sys
import tempfile
import time
from pathlib import Path

import gradio as gr

# Ensure ranker package is importable
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ranker.features import extract_all_features
from ranker.honeypot import detect_honeypot
from ranker.ltr import fallback_weighted_scoring
from ranker.reasoning import generate_reasoning
from ranker.validator import validate_submission


def run_ranking(file_obj, top_n: int = 100):
    """
    Run the ranking pipeline on uploaded candidates.

    Parameters
    ----------
    file_obj : file
        Uploaded JSON or JSONL file containing candidate data.
    top_n : int
        Number of top candidates to return (max 100).

    Returns
    -------
    tuple[str, str, str]
        (summary_text, csv_path, detail_text)
    """
    start_time = time.time()

    if file_obj is None:
        return "❌ Please upload a candidates file.", None, ""

    # Read file content
    file_path = file_obj.name if hasattr(file_obj, 'name') else str(file_obj)
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    # Parse candidates (JSON array or JSONL)
    candidates = []
    if content.startswith("["):
        candidates = json.loads(content)
    else:
        for line in content.split("\n"):
            line = line.strip()
            if line:
                candidates.append(json.loads(line))

    n_total = len(candidates)
    if n_total == 0:
        return "❌ No candidates found in the uploaded file.", None, ""

    if n_total > 10000:
        return f"❌ Too many candidates ({n_total:,}). This demo supports ≤10,000.", None, ""

    # Build candidate lookup
    candidates_by_id = {c["candidate_id"]: c for c in candidates}

    # Stage 1: Feature extraction + honeypot detection
    candidate_features = {}
    honeypot_details = []

    for cand in candidates:
        cid = cand["candidate_id"]
        features = extract_all_features(cand)

        # Honeypot detection
        is_hp, flags = detect_honeypot(cand)
        features["is_honeypot"] = 1.0 if is_hp else 0.0
        features["honeypot_flag_count"] = float(len(flags))
        features["has_fictional_company"] = 1.0 if any(
            f.startswith("FICTIONAL_COMPANY") for f in flags
        ) else 0.0
        features["has_maturity_impossible"] = 1.0 if any(
            f.startswith("MATURITY_IMPOSSIBLE") for f in flags
        ) else 0.0

        # No semantic features for small sample (no FAISS/BM25)
        features["cosine_similarity_jd"] = 0.0
        features["bm25_score_jd"] = 0.0

        candidate_features[cid] = features

        if is_hp:
            honeypot_details.append(
                f"  ⚠ {cid} ({cand['profile']['current_title']}): "
                f"{', '.join(flags[:3])}"
            )

    # Stage 2: Multi-signal scoring
    ranked = fallback_weighted_scoring(candidate_features)

    # Stage 3: Honeypot pruning
    clean_candidates = []
    pruned = 0
    for cid, score in ranked:
        feats = candidate_features.get(cid, {})
        if feats.get("is_honeypot", 0) > 0:
            pruned += 1
            continue
        clean_candidates.append((cid, score))

    # Take top N (max 100)
    top_n = min(top_n, 100, len(clean_candidates))
    final = clean_candidates[:top_n]

    # Stage 4: Score normalization + reasoning
    results = []
    n = len(final)
    for rank_idx, (cid, raw_score) in enumerate(final):
        rank = rank_idx + 1

        # Power curve normalization
        if n > 1:
            t = rank_idx / (n - 1)
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

    # Enforce monotonic scores
    for i in range(1, len(results)):
        if results[i]["score"] >= results[i - 1]["score"]:
            results[i]["score"] = round(results[i - 1]["score"] - 1e-4, 4)

    for r in results:
        r["score"] = max(0.01, r["score"])

    # Write CSV to temp file
    csv_path = tempfile.mktemp(suffix=".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for r in results:
            writer.writerow([
                r["candidate_id"],
                r["rank"],
                f"{r['score']:.4f}",
                r["reasoning"],
            ])

    elapsed = time.time() - start_time

    # Build summary
    summary_lines = [
        f"## Pipeline Complete",
        f"",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Candidates loaded | {n_total:,} |",
        f"| Honeypots detected | {pruned} |",
        f"| Clean candidates | {len(clean_candidates):,} |",
        f"| **Ranked output** | **{len(results)}** |",
        f"| Runtime | {elapsed:.1f}s |",
        f"",
        f"### Top 10 Candidates",
        f"",
        f"| Rank | ID | Title | YoE | Score |",
        f"|---|---|---|---|---|",
    ]

    for r in results[:10]:
        cand = candidates_by_id.get(r["candidate_id"], {})
        title = cand.get("profile", {}).get("current_title", "?")
        yoe = cand.get("profile", {}).get("years_of_experience", 0)
        summary_lines.append(
            f"| #{r['rank']} | {r['candidate_id']} | {title} | {yoe:.1f}y | {r['score']:.4f} |"
        )

    # Detail text
    detail_lines = [f"### Honeypot Analysis ({pruned} flagged)"]
    if honeypot_details:
        detail_lines.extend(honeypot_details[:20])
    else:
        detail_lines.append("  No honeypots detected in this sample.")

    detail_lines.append("")
    detail_lines.append("### Sample Reasoning (Rank 1)")
    if results:
        detail_lines.append(f"  \"{results[0]['reasoning']}\"")

    return "\n".join(summary_lines), csv_path, "\n".join(detail_lines)


# ── Gradio Interface ──

DESCRIPTION = """
# 🧠 Redrob AI Candidate Ranking System

**Hackathon Submission**: India Runs — The Data & AI Challenge

Upload a candidate JSON or JSONL file (≤100 candidates) and get a ranked CSV output.

### How it works:
1. **Feature Extraction** — 50+ structural, skill, behavioral, and location features per candidate
2. **Multi-Signal Scoring** — Additive formula: core fit (50%) + semantic (20%) + behavioral (17%) + availability (7%) + location (6%)
3. **Honeypot Pruning** — 6-layer adversarial detection removes fake/impossible profiles
4. **Reasoning Generation** — Rule-based, fact-grounded 2-sentence assessment per candidate

> ⚡ This demo uses the fallback scoring path (no FAISS/BM25 recall).
> The full pipeline with dense+sparse retrieval runs on the complete 100K dataset.
"""

demo = gr.Interface(
    fn=run_ranking,
    inputs=[
        gr.File(
            label="Upload Candidates (JSON or JSONL)",
            file_types=[".json", ".jsonl"],
        ),
        gr.Slider(
            minimum=10, maximum=100, value=100, step=1,
            label="Top N candidates to rank",
        ),
    ],
    outputs=[
        gr.Markdown(label="Results Summary"),
        gr.File(label="Download Ranked CSV"),
        gr.Textbox(label="Details (Honeypots & Reasoning)", lines=10),
    ],
    title="Redrob AI Candidate Ranker",
    description=DESCRIPTION,
    examples=[],
)


if __name__ == "__main__":
    demo.launch()
