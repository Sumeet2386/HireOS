"""
download_artifacts.py -- Download pre-computed artifacts from HuggingFace.

Run this ONCE before docker build or local inference if you don't have the
artifacts/ directory.

Usage::

    python download_artifacts.py [--out artifacts]

No authentication required -- the repository is public.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

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
    """Download artifacts and verify completeness."""
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
        logger.error("huggingface_hub not installed.")
        logger.error("Install it with: pip install huggingface_hub")
        sys.exit(1)

    logger.info("Downloading artifacts from hf.co/%s -> %s/", REPO_ID, output_dir)
    logger.info("  %d files to download (~560 MB total)", len(EXPECTED_FILES))

    for i, filename in enumerate(EXPECTED_FILES, 1):
        dest = output_dir / filename
        if dest.exists():
            size_mb = dest.stat().st_size / 1e6
            logger.info(
                "  [%d/%d] %s -- already exists (%.1f MB), skipping",
                i, len(EXPECTED_FILES), filename, size_mb,
            )
            continue

        logger.info("  [%d/%d] Downloading %s...", i, len(EXPECTED_FILES), filename)
        try:
            downloaded_path = hf_hub_download(
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                filename=filename,
                local_dir=str(output_dir),
            )
            size_mb = Path(downloaded_path).stat().st_size / 1e6
            logger.info("    OK (%.1f MB)", size_mb)
        except Exception as exc:
            logger.error("    FAILED: %s", exc)
            sys.exit(1)

    # Verify all files
    logger.info("Verifying...")
    missing = [f for f in EXPECTED_FILES if not (output_dir / f).exists()]
    if missing:
        logger.error("Missing files: %s", missing)
        sys.exit(1)

    total_mb = sum((output_dir / f).stat().st_size for f in EXPECTED_FILES) / 1e6
    logger.info("All %d artifacts present (%.0f MB total)", len(EXPECTED_FILES), total_mb)
    logger.info("Ready for: python rank.py --artifacts %s ...", args.out)


if __name__ == "__main__":
    main()
