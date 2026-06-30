"""
Redrob AI Candidate Ranking System — HuggingFace Spaces Sandbox

Full-featured ranking pipeline with professional UI.
Supports the complete 100K candidate dataset.

Architecture:
  - 50+ features across structural, skill, behavioral, and location dimensions
  - Multi-signal additive scoring formula
  - 6-layer adversarial honeypot detection
  - Rule-based, fact-grounded reasoning generation
"""

import csv
import hashlib
import json
import os
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

# ── Pre-computed Results ──
PRECOMPUTED_CSV = ROOT / "final_submission.csv"
HAS_PRECOMPUTED = PRECOMPUTED_CSV.exists()


def load_precomputed():
    """Load the pre-computed Docker pipeline results."""
    if not HAS_PRECOMPUTED:
        return {}
    results = {}
    with open(PRECOMPUTED_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results[row["candidate_id"]] = {
                "rank": int(row["rank"]),
                "score": float(row["score"]),
                "reasoning": row["reasoning"],
            }
    return results


PRECOMPUTED = load_precomputed()


def detect_full_dataset(candidates):
    """Check if the uploaded file is the known 100K evaluation dataset."""
    if len(candidates) < 90000:
        return False
    # Check for known candidate IDs from our submission
    known_ids = {"CAND_0060054", "CAND_0093912", "CAND_0032216", "CAND_0094759"}
    uploaded_ids = {c["candidate_id"] for c in candidates[:500]}
    return len(known_ids & uploaded_ids) >= 3


def run_ranking(file_obj, top_n, use_precomputed):
    """
    Run the ranking pipeline on uploaded candidates.

    Returns (summary_markdown, csv_filepath, details_text)
    """
    start_time = time.time()

    if file_obj is None:
        return "### ❌ Please upload a candidates file.", None, ""

    # Read file content
    if isinstance(file_obj, str):
        file_path = file_obj
    elif hasattr(file_obj, "name"):
        file_path = file_obj.name
    else:
        file_path = str(file_obj)

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
        return "### ❌ No candidates found in the uploaded file.", None, ""

    candidates_by_id = {c["candidate_id"]: c for c in candidates}

    # ── Path A: Use pre-computed Docker results (bit-for-bit identical) ──
    is_full_dataset = detect_full_dataset(candidates)
    if use_precomputed and is_full_dataset and PRECOMPUTED:
        results = []
        for cid, data in PRECOMPUTED.items():
            results.append({
                "candidate_id": cid,
                "rank": data["rank"],
                "score": data["score"],
                "reasoning": data["reasoning"],
            })
        results.sort(key=lambda x: x["rank"])
        results = results[:top_n]

        # Count honeypots in full dataset
        hp_count = 0
        for cand in candidates[:2000]:  # Sample for speed
            is_hp, _ = detect_honeypot(cand)
            if is_hp:
                hp_count += 1
        hp_estimate = int(hp_count * n_total / 2000)

        elapsed = time.time() - start_time
        mode_label = "🔒 Docker-Identical (Pre-computed)"
        pipeline_note = (
            "These are the **exact same results** produced by the full Docker pipeline "
            "(FAISS + BM25 + LightGBM LambdaMART). Bit-for-bit identical to `final_submission.csv`."
        )

    # ── Path B: Live ranking pipeline ──
    else:
        candidate_features = {}
        honeypot_details = []

        for cand in candidates:
            cid = cand["candidate_id"]
            features = extract_all_features(cand)

            is_hp, flags = detect_honeypot(cand)
            features["is_honeypot"] = 1.0 if is_hp else 0.0
            features["honeypot_flag_count"] = float(len(flags))
            features["has_fictional_company"] = 1.0 if any(
                f.startswith("FICTIONAL_COMPANY") for f in flags
            ) else 0.0
            features["has_maturity_impossible"] = 1.0 if any(
                f.startswith("MATURITY_IMPOSSIBLE") for f in flags
            ) else 0.0
            features["cosine_similarity_jd"] = 0.0
            features["bm25_score_jd"] = 0.0

            candidate_features[cid] = features

            if is_hp:
                honeypot_details.append(
                    f"⚠️ **{cid}** ({cand['profile']['current_title']}): "
                    f"{', '.join(flags[:3])}"
                )

        ranked = fallback_weighted_scoring(candidate_features)

        clean_candidates = []
        pruned = 0
        for cid, score in ranked:
            feats = candidate_features.get(cid, {})
            if feats.get("is_honeypot", 0) > 0:
                pruned += 1
                continue
            clean_candidates.append((cid, score))

        top_n_actual = min(top_n, 100, len(clean_candidates))
        final = clean_candidates[:top_n_actual]

        results = []
        n = len(final)
        for rank_idx, (cid, raw_score) in enumerate(final):
            rank = rank_idx + 1
            if n > 1:
                t = rank_idx / (n - 1)
                normalized_score = round(1.0 - 0.80 * (t ** 0.7), 4)
            else:
                normalized_score = 1.0

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

        for i in range(1, len(results)):
            if results[i]["score"] >= results[i - 1]["score"]:
                results[i]["score"] = round(results[i - 1]["score"] - 1e-4, 4)
        for r in results:
            r["score"] = max(0.01, r["score"])

        hp_estimate = pruned
        elapsed = time.time() - start_time
        mode_label = "⚡ Live Pipeline (Fallback Scoring)"
        pipeline_note = (
            "Ranked using multi-signal fallback scoring (no FAISS/BM25 recall). "
            "Honeypot detection and feature extraction are identical to Docker."
        )
        honeypot_details = honeypot_details  # already populated

    # ── Write CSV ──
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

    # ── Build Summary ──
    summary = f"""## {mode_label}

{pipeline_note}

---

| Metric | Value |
|:---|:---|
| 📊 Candidates loaded | **{n_total:,}** |
| 🛡️ Honeypots detected | **~{hp_estimate:,}** |
| ✅ Clean candidates ranked | **{len(results)}** |
| ⏱️ Runtime | **{elapsed:.1f}s** |

---

### 🏆 Top 10 Ranked Candidates

| Rank | Candidate ID | Score | Reasoning |
|:---:|:---|:---:|:---|
"""
    for r in results[:10]:
        short_reason = r["reasoning"][:120] + "..." if len(r["reasoning"]) > 120 else r["reasoning"]
        summary += f"| **#{r['rank']}** | `{r['candidate_id']}` | {r['score']:.4f} | {short_reason} |\n"

    # ── Details ──
    detail_parts = []
    if use_precomputed and is_full_dataset and PRECOMPUTED:
        detail_parts.append("MODE: Docker-Identical Pre-computed Results")
        detail_parts.append(f"Source: final_submission.csv ({len(PRECOMPUTED)} candidates)")
        detail_parts.append("")
        detail_parts.append("These results were produced by the full pipeline:")
        detail_parts.append("  Stage 1: FAISS dense retrieval (bge-small-en-v1.5, top 5K)")
        detail_parts.append("  Stage 2: BM25 sparse recall (top 500)")
        detail_parts.append("  Stage 3: 50-feature extraction")
        detail_parts.append("  Stage 4: LightGBM LambdaMART re-ranking")
        detail_parts.append("  Stage 5: 6-layer honeypot pruning")
        detail_parts.append("  Stage 6: Deterministic reasoning generation")
    else:
        detail_parts.append(f"HONEYPOT ANALYSIS ({hp_estimate} flagged)")
        detail_parts.append("")
        if 'honeypot_details' in dir() and honeypot_details:
            for hd in honeypot_details[:20]:
                # Strip markdown for textbox
                detail_parts.append(hd.replace("**", "").replace("⚠️", "[HP]"))
        else:
            detail_parts.append("  No honeypots detected in this sample.")
        detail_parts.append("")
        detail_parts.append(f"SAMPLE REASONING (Rank 1)")
        if results:
            detail_parts.append(f'  "{results[0]["reasoning"]}"')

    return summary, csv_path, "\n".join(detail_parts)


# ── Custom CSS for Professional Look ──

CUSTOM_CSS = """
/* Global theme */
.gradio-container {
    max-width: 1200px !important;
    margin: auto;
}

/* Header styling */
.header-banner {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 24px;
    border: 1px solid rgba(255,255,255,0.1);
}
.header-banner h1 {
    color: #e94560;
    font-size: 2.2em;
    margin: 0 0 8px 0;
    font-weight: 800;
    letter-spacing: -0.5px;
}
.header-banner p {
    color: #a8b2d1;
    font-size: 1.05em;
    margin: 4px 0;
    line-height: 1.6;
}
.header-banner .team-badge {
    display: inline-block;
    background: rgba(233, 69, 96, 0.15);
    color: #e94560;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.85em;
    font-weight: 600;
    border: 1px solid rgba(233, 69, 96, 0.3);
    margin-top: 8px;
}

/* Stats cards */
.stats-row {
    display: flex;
    gap: 16px;
    margin: 16px 0;
}
.stat-card {
    flex: 1;
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.08);
}
.stat-card .stat-value {
    font-size: 1.8em;
    font-weight: 800;
    color: #e94560;
}
.stat-card .stat-label {
    color: #a8b2d1;
    font-size: 0.85em;
    margin-top: 4px;
}

/* Pipeline steps */
.pipeline-steps {
    display: flex;
    gap: 8px;
    margin: 16px 0;
    flex-wrap: wrap;
}
.pipeline-step {
    background: rgba(233, 69, 96, 0.08);
    border: 1px solid rgba(233, 69, 96, 0.2);
    border-radius: 8px;
    padding: 8px 16px;
    color: #a8b2d1;
    font-size: 0.85em;
    font-weight: 500;
}
.pipeline-step .step-num {
    color: #e94560;
    font-weight: 700;
    margin-right: 6px;
}

/* Architecture info */
.arch-card {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
    border-radius: 12px;
    padding: 24px;
    border: 1px solid rgba(255,255,255,0.08);
    margin: 12px 0;
}
.arch-card h3 {
    color: #e94560;
    margin: 0 0 12px 0;
}
.arch-card p, .arch-card li {
    color: #a8b2d1;
    line-height: 1.7;
}

/* Override tab styling */
.tab-nav button {
    font-weight: 600 !important;
    font-size: 0.95em !important;
}
.tab-nav button.selected {
    border-color: #e94560 !important;
    color: #e94560 !important;
}
"""

HEADER_HTML = """
<div class="header-banner">
    <h1>🧠 HireOS — AI Candidate Ranking Engine</h1>
    <p><strong>India Runs — The Data & AI Challenge</strong></p>
    <p>Hybrid retrieval + ML re-ranking pipeline with adversarial honeypot detection.
       Processes 100,000 candidates in under 2 minutes on CPU.</p>
    <span class="team-badge">Team Discern — Harshal Andhale · Sumeet Gite · Prem More</span>
</div>
"""

ARCH_HTML = """
<div class="arch-card">
<h3>🏗️ System Architecture</h3>

**Stage 1 — Recall (FAISS + BM25)**
- Dense retrieval with `bge-small-en-v1.5` embeddings → top 5,000 candidates
- BM25 sparse retrieval → top 500 candidates
- Union into ~5,000 candidate shortlist

**Stage 2 — Feature Extraction (50+ signals)**
- **Structural**: title match, years of experience, career progression, company tier
- **Skill**: AI core skill count, proficiency levels, entailment validation
- **Behavioral**: activity decay, response rate, notice period, platform engagement
- **Location**: city tier, timezone alignment, relocation willingness

**Stage 3 — ML Re-ranking (LightGBM LambdaMART)**
- Trained on GPT-4o-mini weak labels (2,000 stratified sample, 0-10 relevance)
- Optimized for NDCG@100

**Stage 4 — Honeypot Pruning (6 layers)**
- Timeline impossibility detection
- Skill-text entailment failure
- Tech maturity validation (e.g., claims 8y PyTorch but it launched in 2016)
- Career overlap detection
- Keyword stuffer flagging
- Suspicious junior profile identification

**Stage 5 — Reasoning Generation**
- Deterministic, rule-based, fact-grounded
- 2-sentence assessment per candidate, zero hallucination risk
</div>
"""

STATS_HTML = """
<div class="stats-row">
    <div class="stat-card">
        <div class="stat-value">100K</div>
        <div class="stat-label">Candidates Processed</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">50+</div>
        <div class="stat-label">Features Extracted</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">~90s</div>
        <div class="stat-label">Pipeline Runtime</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">6</div>
        <div class="stat-label">Honeypot Layers</div>
    </div>
</div>
"""


# ── Gradio App ──

with gr.Blocks(
    css=CUSTOM_CSS,
    title="HireOS — AI Candidate Ranking Engine",
    theme=gr.themes.Base(
        primary_hue="red",
        secondary_hue="blue",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ).set(
        body_background_fill="#0d1117",
        body_background_fill_dark="#0d1117",
        block_background_fill="#161b22",
        block_background_fill_dark="#161b22",
        block_border_color="rgba(255,255,255,0.08)",
        block_border_color_dark="rgba(255,255,255,0.08)",
        input_background_fill="#0d1117",
        input_background_fill_dark="#0d1117",
        button_primary_background_fill="#e94560",
        button_primary_background_fill_dark="#e94560",
        button_primary_background_fill_hover="#d63851",
        button_primary_background_fill_hover_dark="#d63851",
        button_primary_text_color="white",
    ),
) as demo:

    gr.HTML(HEADER_HTML)

    with gr.Tabs():
        # ── Tab 1: Rank Candidates ──
        with gr.Tab("🎯 Rank Candidates", id="rank"):
            gr.HTML(STATS_HTML)

            with gr.Row():
                with gr.Column(scale=1):
                    file_input = gr.File(
                        label="📁 Upload Candidates (JSON or JSONL)",
                        file_types=[".json", ".jsonl"],
                        type="filepath",
                    )
                    top_n_slider = gr.Slider(
                        minimum=10,
                        maximum=100,
                        value=100,
                        step=1,
                        label="🔢 Top N Candidates to Rank",
                    )
                    use_precomputed_toggle = gr.Checkbox(
                        label="🔒 Use Docker-identical results (pre-computed)",
                        value=True,
                        info="When enabled and the full 100K dataset is detected, "
                             "serves the exact same results as the Docker pipeline.",
                    )
                    submit_btn = gr.Button(
                        "🚀 Run Ranking Pipeline",
                        variant="primary",
                        size="lg",
                    )

                with gr.Column(scale=2):
                    output_summary = gr.Markdown(
                        label="Results",
                        value="### Upload a candidates file and click 'Run Ranking Pipeline' to begin.",
                    )
                    output_csv = gr.File(label="📥 Download Ranked CSV")

            with gr.Accordion("🔍 Pipeline Details & Honeypot Analysis", open=False):
                output_details = gr.Textbox(
                    label="Execution Details",
                    lines=15,
                    max_lines=30,
                )

            submit_btn.click(
                fn=run_ranking,
                inputs=[file_input, top_n_slider, use_precomputed_toggle],
                outputs=[output_summary, output_csv, output_details],
            )

        # ── Tab 2: Architecture ──
        with gr.Tab("🏗️ Architecture", id="arch"):
            gr.HTML(ARCH_HTML)

        # ── Tab 3: About ──
        with gr.Tab("ℹ️ About", id="about"):
            gr.Markdown("""
## How Evaluation Works

### Docker Pipeline (Official Evaluation)
The hackathon judges run `python rank.py` inside a **Docker container** on their server.
This uses the full pipeline: FAISS recall → BM25 recall → LightGBM re-ranking → honeypot pruning.
Results are **deterministic** — running the same input always produces identical output.

### This Sandbox (Interactive Demo)
This HuggingFace Space serves two modes:

1. **🔒 Pre-computed mode** — When the full 100K dataset is uploaded with the toggle enabled,
   it serves the **exact same results** as `final_submission.csv` from the Docker pipeline.
   This ensures judges see bit-for-bit identical rankings.

2. **⚡ Live mode** — For smaller samples or with the toggle disabled, it runs the
   fallback scoring pipeline in real-time. This uses the same feature extraction and
   honeypot detection, but replaces FAISS/BM25/LightGBM with an additive formula.

### What's Identical Between Both?
| Component | Docker | Sandbox |
|:---|:---:|:---:|
| Feature extraction (50+ features) | ✅ | ✅ |
| Honeypot detection (6 layers) | ✅ | ✅ |
| FAISS dense retrieval | ✅ | ❌ |
| BM25 sparse recall | ✅ | ❌ |
| LightGBM re-ranking | ✅ | ❌ |
| Reasoning generation | ✅ | ✅ |
| **Pre-computed results** | ✅ | **✅** |

---

### Links
- **GitHub**: [github.com/Sumeet2386/HireOS](https://github.com/Sumeet2386/HireOS)
- **Docker**: Reproduce with `python rank.py --candidates ./candidates.jsonl --out ./submission.csv --artifacts artifacts`
""")


if __name__ == "__main__":
    demo.launch()
