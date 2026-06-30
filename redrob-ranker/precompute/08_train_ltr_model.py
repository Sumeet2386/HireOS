"""
08_train_ltr_model.py — Train a LightGBM LambdaMART model for Learning-to-Rank.

Uses weak labels from GPT-4o-mini + pre-computed features to train a
LambdaMART model that optimizes NDCG@100.

Usage:
    python precompute/08_train_ltr_model.py --features <parquet> --labels <json> --out <output_dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LightGBM LTR model.")
    parser.add_argument("--features", default="artifacts/features.parquet",
                        help="Path to features.parquet")
    parser.add_argument("--labels", default="artifacts/weak_labels.json",
                        help="Path to weak_labels.json")
    parser.add_argument("--out", default="artifacts", help="Output directory")
    parser.add_argument("--n-estimators", type=int, default=200, help="Number of boosting rounds")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--max-depth", type=int, default=6, help="Max tree depth")
    args = parser.parse_args()

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load weak labels
    print("Loading weak labels...")
    with open(args.labels, "r", encoding="utf-8") as f:
        labels_data = json.load(f)

    label_map = {item["candidate_id"]: item["weak_label"] for item in labels_data}
    print(f"  {len(label_map)} labeled candidates")

    # Load features
    print("Loading features...")
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(args.features)
        df = table.to_pydict()
    except ImportError:
        import csv
        df = {}
        with open(args.features.replace(".parquet", ".csv"), "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for key, val in row.items():
                    if key not in df:
                        df[key] = []
                    df[key].append(val)

    cids = df.pop("candidate_id")
    feature_columns = sorted(df.keys())
    print(f"  {len(cids)} candidates, {len(feature_columns)} features")
    print(f"  Feature columns: {feature_columns}")

    # Filter to labeled candidates only
    labeled_indices = []
    labels = []
    for i, cid in enumerate(cids):
        if cid in label_map:
            labeled_indices.append(i)
            labels.append(label_map[cid])

    print(f"  {len(labeled_indices)} candidates have labels (matched)")

    if len(labeled_indices) < 100:
        print("ERROR: Too few labeled candidates. Need at least 100.")
        sys.exit(1)

    # Build feature matrix
    n = len(labeled_indices)
    m = len(feature_columns)
    X = np.zeros((n, m), dtype=np.float32)
    y = np.array(labels, dtype=np.float32)

    for row_out, row_in in enumerate(labeled_indices):
        for j, col in enumerate(feature_columns):
            try:
                X[row_out, j] = float(df[col][row_in])
            except (ValueError, IndexError):
                X[row_out, j] = 0.0

    print(f"\nTraining data: X={X.shape}, y={y.shape}")
    print(f"  Label distribution: min={y.min():.1f}, max={y.max():.1f}, "
          f"mean={y.mean():.2f}, std={y.std():.2f}")

    # Train/validation split (80/20)
    np.random.seed(42)
    indices = np.random.permutation(n)
    split = int(0.8 * n)
    train_idx = indices[:split]
    val_idx = indices[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # For LambdaMART: all candidates belong to one query group
    # (single JD → single query)
    train_group = [len(train_idx)]
    val_group = [len(val_idx)]

    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")

    # Create LightGBM datasets
    train_data = lgb.Dataset(
        X_train, label=y_train,
        group=train_group,
        feature_name=feature_columns,
        free_raw_data=False,
    )
    val_data = lgb.Dataset(
        X_val, label=y_val,
        group=val_group,
        feature_name=feature_columns,
        reference=train_data,
        free_raw_data=False,
    )

    # LambdaMART parameters
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [10, 50, 100],
        "learning_rate": args.lr,
        "max_depth": args.max_depth,
        "num_leaves": 2 ** args.max_depth - 1,
        "min_child_samples": 10,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "verbose": 1,
    }

    print(f"\nTraining LambdaMART with {args.n_estimators} rounds...")
    callbacks = [
        lgb.log_evaluation(period=20),
        lgb.early_stopping(stopping_rounds=30),
    ]

    model = lgb.train(
        params,
        train_data,
        num_boost_round=args.n_estimators,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    print(f"\nBest iteration: {model.best_iteration}")

    # Feature importance
    importance = model.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feature_columns, importance), key=lambda x: -x[1])

    print(f"\nTop 15 features by importance (gain):")
    for name, imp in feat_imp[:15]:
        print(f"  {name:40s} {imp:10.1f}")

    # Save model
    model_path = output_dir / "lgbm_ltr_model.bin"
    model.save_model(str(model_path))
    print(f"\nModel saved to {model_path}")

    # Save feature columns for inference
    cols_path = output_dir / "feature_columns.json"
    with open(cols_path, "w", encoding="utf-8") as f:
        json.dump(feature_columns, f)
    print(f"Feature columns saved to {cols_path}")

    # Save feature importance
    imp_path = output_dir / "feature_importance.json"
    with open(imp_path, "w", encoding="utf-8") as f:
        json.dump(feat_imp, f, indent=2)
    print(f"Feature importance saved to {imp_path}")

    # Validation: predict on val set and compute basic stats
    val_preds = model.predict(X_val)
    print(f"\nValidation predictions:")
    print(f"  Predicted: min={val_preds.min():.3f}, max={val_preds.max():.3f}, "
          f"mean={val_preds.mean():.3f}")
    print(f"  Actual:    min={y_val.min():.1f}, max={y_val.max():.1f}, "
          f"mean={y_val.mean():.2f}")

    # Spearman correlation
    from scipy.stats import spearmanr
    corr, pval = spearmanr(val_preds, y_val)
    print(f"  Spearman correlation: {corr:.4f} (p={pval:.4e})")


if __name__ == "__main__":
    main()
