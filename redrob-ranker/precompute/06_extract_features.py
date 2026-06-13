"""
06_extract_features.py — Extract structural features for ALL candidates → Parquet.

This pre-computes all features that don't depend on the JD query (structural,
behavioral, location, honeypot) so the online phase only needs to add
semantic features.

Usage:
    python precompute/06_extract_features.py --candidates <path> --out <output_dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Add parent to path for ranker imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ranker.features import extract_all_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract features to Parquet.")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--out", default="artifacts", help="Output directory")
    args = parser.parse_args()

    input_path = Path(args.candidates)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_ids: list[str] = []
    all_features: list[dict[str, float]] = []

    total = 0
    honeypot_count = 0

    print(f"Reading {input_path}...")

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            candidate = json.loads(line)

            cid = candidate.get("candidate_id", "")
            features = extract_all_features(candidate)

            candidate_ids.append(cid)
            all_features.append(features)

            if features.get("is_honeypot", 0) > 0:
                honeypot_count += 1

            if total % 10000 == 0:
                print(f"  Processed {total:,} candidates... ({honeypot_count} honeypots detected)")

    print(f"\nTotal: {total:,} candidates")
    print(f"Honeypots detected: {honeypot_count:,} ({honeypot_count/total*100:.1f}%)")

    # Determine feature columns
    feature_columns = sorted(all_features[0].keys())
    print(f"Feature columns ({len(feature_columns)}): {feature_columns}")

    # Save as Parquet
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        columns = {"candidate_id": candidate_ids}
        for col in feature_columns:
            columns[col] = [f.get(col, 0.0) for f in all_features]

        table = pa.table(columns)
        output_path = output_dir / "features.parquet"
        pq.write_table(table, output_path)
        size_mb = output_path.stat().st_size / 1e6
        print(f"Saved features to {output_path} ({size_mb:.1f} MB)")
    except ImportError:
        # Fallback to CSV
        import csv
        output_path = output_dir / "features.csv"
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["candidate_id"] + feature_columns)
            for cid, feats in zip(candidate_ids, all_features):
                writer.writerow([cid] + [feats.get(col, 0.0) for col in feature_columns])
        print(f"Saved features to {output_path}")

    # Also save feature column names for the online phase
    columns_path = output_dir / "feature_columns.json"
    with open(columns_path, "w") as f:
        json.dump(feature_columns, f)
    print(f"Feature columns saved to {columns_path}")


if __name__ == "__main__":
    main()
