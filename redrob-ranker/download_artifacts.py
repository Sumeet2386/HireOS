"""
download_artifacts.py — Download pre-computed artifacts from HuggingFace.

Run this ONCE before docker build or local inference if you don't have the
artifacts/ directory.

Usage:
    python download_artifacts.py [--out artifacts]

No authentication required — the repository is public.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ID = "harshal9657/HireOS-artifacts"
REPO_TYPE = "model"

# All expected artifact files
EXPECTED_FILES = [
    "bm25_index.pkl",
    "candidate_texts.parquet",
    "eda_stats.json",
    "embeddings.npy",
    "features.parquet",
    "feature_columns.json",
    "feature_importance.json",
    "id_mapping.json",
    "index.faiss",
    "jd_embedding.npy",
    "lgbm_ltr_model.bin",
    "weak_labels.json",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download pre-computed artifacts from HuggingFace."
    )
    parser.add_argument(
        "--out", default="artifacts", help="Output directory (default: artifacts)"
    )
    args = parser.parse_args()

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: huggingface_hub not installed.")
        print("Install it with: pip install huggingface_hub")
        sys.exit(1)

    print(f"Downloading artifacts from hf.co/{REPO_ID} → {output_dir}/")
    print(f"  {len(EXPECTED_FILES)} files to download (~560 MB total)\n")

    for i, filename in enumerate(EXPECTED_FILES, 1):
        dest = output_dir / filename
        if dest.exists():
            size_mb = dest.stat().st_size / 1e6
            print(f"  [{i}/{len(EXPECTED_FILES)}] {filename} — already exists ({size_mb:.1f} MB), skipping")
            continue

        print(f"  [{i}/{len(EXPECTED_FILES)}] Downloading {filename}...", end="", flush=True)
        try:
            downloaded_path = hf_hub_download(
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                filename=filename,
                local_dir=str(output_dir),
            )
            size_mb = Path(downloaded_path).stat().st_size / 1e6
            print(f" OK ({size_mb:.1f} MB)")
        except Exception as e:
            print(f" FAILED: {e}")
            sys.exit(1)

    # Verify all files
    print("\nVerifying...")
    missing = [f for f in EXPECTED_FILES if not (output_dir / f).exists()]
    if missing:
        print(f"ERROR: Missing files: {missing}")
        sys.exit(1)

    total_mb = sum((output_dir / f).stat().st_size for f in EXPECTED_FILES) / 1e6
    print(f"All {len(EXPECTED_FILES)} artifacts present ({total_mb:.0f} MB total)")
    print("Ready for: python rank.py --artifacts artifacts ...")


if __name__ == "__main__":
    main()
