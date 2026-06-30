"""
09_precompute_cross_encoder.py — Precompute cross-encoder reranking scores.

Scores the top N candidates (from the recall phase) against the JD using
a cross-encoder model and saves the results.  These precomputed scores
are loaded during inference for zero-latency reranking.

Usage:
    python precompute/09_precompute_cross_encoder.py \
        --candidates <jsonl> \
        --artifacts <dir> \
        [--model mixedbread-ai/mxbai-rerank-xsmall-v1] \
        [--top-n 500]

Requires: sentence-transformers, torch
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


JD_FULL_TEXT = (
    "Senior AI Engineer — Founding Team at Redrob AI, a Series A AI-native "
    "talent intelligence platform. Location: Pune/Noida, India (Hybrid). "
    "Experience: 5-9 years. Building the intelligence layer: ranking, "
    "retrieval, and matching systems. "
    "Must have production experience with embeddings-based retrieval "
    "(sentence-transformers, BGE, E5), vector databases or hybrid search "
    "(Pinecone, Weaviate, Qdrant, Milvus, FAISS), strong Python, "
    "and hands-on experience designing evaluation frameworks for ranking "
    "systems (NDCG, MRR, MAP, A/B testing). "
    "Nice to have: LLM fine-tuning (LoRA, QLoRA), learning-to-rank models, "
    "HR-tech or marketplace experience, distributed systems. "
    "Do NOT want: title chasers, framework enthusiasts, only-consulting "
    "careers (TCS, Infosys, Wipro), primary CV/Speech/Robotics without "
    "NLP/IR exposure. "
    "India-based preferred (Pune, Noida, Hyderabad, Mumbai, Delhi NCR). "
    "Sub-30-day notice period preferred. Looking for a shipper over a "
    "researcher — someone who builds, deploys, and iterates in production."
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute cross-encoder reranking scores."
    )
    parser.add_argument(
        "--candidates", required=True,
        help="Path to candidates.jsonl",
    )
    parser.add_argument(
        "--artifacts", default="artifacts",
        help="Artifacts directory (reads recall artifacts, writes scores)",
    )
    parser.add_argument(
        "--model", default="mixedbread-ai/mxbai-rerank-xsmall-v1",
        help="Cross-encoder model name",
    )
    parser.add_argument(
        "--top-n", type=int, default=500,
        help="Number of top recall candidates to score",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for cross-encoder inference",
    )
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts)

    # ---- Step 1: Get recall candidate list ----
    print("Loading recall artifacts to identify top candidates...")
    t0 = time.time()

    # Load FAISS index + JD embedding for recall
    import faiss
    faiss_index = faiss.read_index(str(artifacts_dir / "index.faiss"))
    jd_embedding = np.load(artifacts_dir / "jd_embedding.npy").astype(np.float32)
    with open(artifacts_dir / "id_mapping.json", "r", encoding="utf-8") as f:
        id_mapping = {int(k): v for k, v in json.load(f).items()}

    # Run dense recall to get top candidates
    if jd_embedding.ndim == 1:
        jd_embedding = jd_embedding.reshape(1, -1)
    scores, indices = faiss_index.search(jd_embedding, args.top_n)
    recall_cids = []
    for idx in indices[0]:
        if idx >= 0:
            cid = id_mapping.get(int(idx))
            if cid:
                recall_cids.append(cid)

    print(f"  Identified {len(recall_cids)} top candidates via dense recall")

    # ---- Step 2: Load candidate records ----
    print("Loading candidate records...")
    recall_set = set(recall_cids)
    candidates = {}
    with open(args.candidates, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            cand = json.loads(line)
            cid = cand["candidate_id"]
            if cid in recall_set:
                candidates[cid] = cand

    print(f"  Loaded {len(candidates)} candidate records")

    # ---- Step 3: Cross-encoder scoring ----
    print(f"\nScoring with cross-encoder '{args.model}'...")
    print(f"  Candidates: {len(candidates)}")
    print(f"  Batch size: {args.batch_size}")

    from ranker.reranker import rerank_with_cross_encoder

    candidate_pairs = [(cid, candidates[cid]) for cid in recall_cids if cid in candidates]

    t_ce = time.time()
    results = rerank_with_cross_encoder(
        JD_FULL_TEXT,
        candidate_pairs,
        model_name=args.model,
        batch_size=args.batch_size,
    )
    ce_time = time.time() - t_ce

    print(f"\n  Cross-encoder scoring completed in {ce_time:.1f}s")
    print(f"  Throughput: {len(results) / ce_time:.1f} candidates/sec")

    # ---- Step 4: Save scores ----
    scores_dict = {cid: float(score) for cid, score in results}
    output_path = artifacts_dir / "cross_encoder_scores.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scores_dict, f, indent=2)

    print(f"\n  Saved {len(scores_dict)} cross-encoder scores to {output_path}")

    # Print score distribution
    score_vals = list(scores_dict.values())
    print(f"\n  Score distribution:")
    print(f"    Min:    {min(score_vals):.4f}")
    print(f"    Max:    {max(score_vals):.4f}")
    print(f"    Mean:   {np.mean(score_vals):.4f}")
    print(f"    Median: {np.median(score_vals):.4f}")
    print(f"    Std:    {np.std(score_vals):.4f}")

    total_time = time.time() - t0
    print(f"\n  Total precompute time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
