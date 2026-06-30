"""
HireOS — AI Candidate Ranking Engine
HuggingFace Spaces Sandbox for India Runs Data & AI Challenge
"""

import csv
import json
import sys
import tempfile
import time
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ranker.features import extract_all_features
from ranker.honeypot import detect_honeypot
from ranker.ltr import fallback_weighted_scoring
from ranker.reasoning import generate_reasoning

# ── Pre-computed Docker results ──
PRECOMPUTED_CSV = ROOT / "team_Discern.csv"


def load_precomputed():
    if not PRECOMPUTED_CSV.exists():
        return {}
    results = {}
    with open(PRECOMPUTED_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            results[row["candidate_id"]] = {
                "rank": int(row["rank"]),
                "score": float(row["score"]),
                "reasoning": row["reasoning"],
            }
    return results


PRECOMPUTED = load_precomputed()


def detect_full_dataset(candidates):
    if len(candidates) < 90000:
        return False
    known = {"CAND_0060054", "CAND_0093912", "CAND_0032216", "CAND_0094759"}
    uploaded = {c["candidate_id"] for c in candidates[:500]}
    return len(known & uploaded) >= 3


def run_ranking(file_obj, top_n, use_precomputed):
    t0 = time.time()

    if file_obj is None:
        return "⬆️ Upload a candidates file to get started.", None, ""

    file_path = file_obj if isinstance(file_obj, str) else (
        file_obj.name if hasattr(file_obj, "name") else str(file_obj)
    )

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

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
        return "No candidates found in the file.", None, ""

    candidates_by_id = {c["candidate_id"]: c for c in candidates}
    is_full = detect_full_dataset(candidates)

    # ── Pre-computed path ──
    if use_precomputed and is_full and PRECOMPUTED:
        results = sorted(PRECOMPUTED.values(), key=lambda x: x["rank"])
        results = [
            {**r, "candidate_id": cid}
            for cid, r in sorted(PRECOMPUTED.items(), key=lambda x: x[1]["rank"])
        ][:top_n]

        hp_count = sum(1 for c in candidates[:2000] if detect_honeypot(c)[0])
        hp_est = int(hp_count * n_total / 2000)
        elapsed = time.time() - t0
        mode = "Pre-computed (Docker-identical)"
        details = (
            f"Mode: {mode}\n"
            f"Source: team_Discern.csv ({len(PRECOMPUTED)} ranked candidates)\n\n"
            "Full pipeline: FAISS → BM25 → LightGBM LambdaMART → Honeypot pruning → Reasoning"
        )

    # ── Live pipeline path ──
    else:
        candidate_features = {}
        honeypot_lines = []

        for cand in candidates:
            cid = cand["candidate_id"]
            feats = extract_all_features(cand)
            is_hp, flags = detect_honeypot(cand)
            feats["is_honeypot"] = 1.0 if is_hp else 0.0
            feats["honeypot_flag_count"] = float(len(flags))
            feats["has_fictional_company"] = 1.0 if any(
                f.startswith("FICTIONAL_COMPANY") for f in flags) else 0.0
            feats["has_maturity_impossible"] = 1.0 if any(
                f.startswith("MATURITY_IMPOSSIBLE") for f in flags) else 0.0
            feats["cosine_similarity_jd"] = 0.0
            feats["bm25_score_jd"] = 0.0
            candidate_features[cid] = feats
            if is_hp:
                honeypot_lines.append(
                    f"  {cid} ({cand['profile']['current_title']}): {', '.join(flags[:3])}"
                )

        ranked = fallback_weighted_scoring(candidate_features)
        clean = [(cid, s) for cid, s in ranked
                 if candidate_features[cid].get("is_honeypot", 0) == 0]
        pruned = len(ranked) - len(clean)
        final = clean[:min(top_n, 100)]

        results = []
        n = len(final)
        for i, (cid, _) in enumerate(final):
            rank = i + 1
            score = round(1.0 - 0.80 * ((i / max(n - 1, 1)) ** 0.7), 4) if n > 1 else 1.0
            cand = candidates_by_id.get(cid)
            reason = generate_reasoning(cand, rank, score,
                                        features=candidate_features.get(cid)) if cand else ""
            results.append({"candidate_id": cid, "rank": rank, "score": score, "reasoning": reason})

        for i in range(1, len(results)):
            if results[i]["score"] >= results[i - 1]["score"]:
                results[i]["score"] = round(results[i - 1]["score"] - 1e-4, 4)
        for r in results:
            r["score"] = max(0.01, r["score"])

        hp_est = pruned
        elapsed = time.time() - t0
        mode = "Live pipeline (fallback scoring)"
        details = f"Mode: {mode}\nHoneypots flagged: {pruned}\n\n"
        if honeypot_lines:
            details += "Honeypot samples:\n" + "\n".join(honeypot_lines[:15])
        if results:
            details += f"\n\nTop reasoning:\n  \"{results[0]['reasoning']}\""

    # ── CSV output ──
    csv_path = tempfile.mktemp(suffix=".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for r in results:
            w.writerow([r["candidate_id"], r["rank"], f"{r['score']:.4f}", r["reasoning"]])

    # ── Summary ──
    summary = f"""**{mode}** · {n_total:,} candidates · {hp_est:,} honeypots · {elapsed:.1f}s

| Rank | Candidate | Score | Reasoning |
|:---:|:---|:---:|:---|
"""
    for r in results[:10]:
        short = r["reasoning"][:100] + "…" if len(r["reasoning"]) > 100 else r["reasoning"]
        summary += f"| {r['rank']} | {r['candidate_id']} | {r['score']:.4f} | {short} |\n"

    if len(results) > 10:
        summary += f"\n*...and {len(results) - 10} more in the downloadable CSV.*"

    return summary, csv_path, details


# ── App ──

demo = gr.Blocks(
    title="HireOS — AI Candidate Ranker",
    theme=gr.themes.Soft(
        primary_hue="indigo",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ),
)

with demo:
    gr.Markdown(
        "# HireOS — AI Candidate Ranking Engine\n"
        "**India Runs — The Data & AI Challenge** · Team Discern\n\n"
        "Upload candidates → get ranked results with honeypot detection and per-candidate reasoning."
    )

    with gr.Row():
        with gr.Column(scale=1):
            file_in = gr.File(
                label="Candidates file (.json / .jsonl)",
                file_types=[".json", ".jsonl"],
                type="filepath",
            )
            top_n = gr.Slider(10, 100, value=100, step=1, label="Top N to rank")
            precomp = gr.Checkbox(
                label="Use pre-computed Docker results",
                value=True,
                info="Serves identical results to the Docker pipeline for the full 100K dataset.",
            )
            btn = gr.Button("Run ranking", variant="primary")

        with gr.Column(scale=2):
            out_summary = gr.Markdown(value="Results will appear here after you upload and submit.")
            out_csv = gr.File(label="Download CSV")

    with gr.Accordion("Execution details", open=False):
        out_details = gr.Textbox(lines=12)

    btn.click(run_ranking, [file_in, top_n, precomp], [out_summary, out_csv, out_details])

    with gr.Accordion("About the pipeline", open=False):
        gr.Markdown("""
**5-stage pipeline** processing 100K candidates in ~90s on CPU:

1. **Recall** — FAISS dense retrieval (bge-small-en-v1.5) + BM25 sparse retrieval
2. **Features** — 50+ signals: structural, skill, behavioral, location
3. **Re-ranking** — LightGBM LambdaMART trained on GPT-4o-mini weak labels
4. **Honeypot pruning** — 6-layer adversarial detection (timeline, entailment, maturity, overlap, stuffer, junior)
5. **Reasoning** — Deterministic, fact-grounded, 2-sentence assessment per candidate

The sandbox offers two modes: **pre-computed** (bit-for-bit identical to Docker) and **live** (fallback scoring for smaller samples).

[GitHub](https://github.com/Sumeet2386/HireOS) · Reproduce: `python rank.py --candidates ./candidates.jsonl --out ./submission.csv --artifacts artifacts`
""")

if __name__ == "__main__":
    demo.launch()
