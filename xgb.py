"""
XGBoost training/evaluation across multiple fingerprint feature sets.

Expected input files (CSV):
  - {BASE_PREFIX}_x_train_<fp>.csv
  - {BASE_PREFIX}_x_test_<fp>.csv
  - {BASE_PREFIX}_y_train.csv
  - {BASE_PREFIX}_y_test.csv

Each x_*.csv is read with index_col=0 (first column treated as index).
Outputs:
  - Per-run train/test predicted probabilities saved under:
      Prob_Inflampred_external/Prob_<timestamp>/
  - Raw metrics per run:
      {BASE_PREFIX}_XGB_fingerprint_metrics_raw.csv
  - Mean±SD summary table:
      {BASE_PREFIX}_XGB_fingerprint_metrics.csv
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from xgboost import XGBClassifier
except ImportError as e:
    raise SystemExit("XGBoost is not installed. Install with: pip install xgboost") from e

from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)

# ===== User-configurable =====
BASE_PREFIX = "AISMPred"  # Change this to match your file prefix


METRIC_KEYS = [
    "Accuracy",
    "Balanced Accuracy",
    "AUROC",
    "AUPRC",
    "MCC",
    "Precision",
    "Sensitivity",
    "Specificity",
    "F1",
]


@dataclass
class TrainResult:
    metrics: Dict[str, float]
    y_prob_train: np.ndarray
    y_prob_test: np.ndarray
    y_train_true: np.ndarray
    y_test_true: np.ndarray


def _to_1d_int(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y).ravel()
    # Keep as numeric; convert to int when possible
    return pd.to_numeric(y, errors="coerce").astype("Int64").to_numpy()


def train_xgboost(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    *,
    n_estimators: int = 500,
    max_depth: int = 6,
    random_state: int = 42,
    n_jobs: int = -1,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.1,
    reg_lambda: float = 1.0,
    gamma: float = 0.1,
    min_child_weight: float = 1.0,
    verbose: bool = True,
) -> TrainResult:
    """Train an XGBoost binary classifier and compute common metrics on the test set."""
    x_train = np.asarray(x_train)
    x_test = np.asarray(x_test)
    y_train = _to_1d_int(y_train)
    y_test = _to_1d_int(y_test)

    # Handle imbalance via scale_pos_weight
    n_pos = np.nansum(y_train == 1)
    n_neg = np.nansum(y_train == 0)
    scale_pos_weight = float(n_neg) / float(n_pos) if n_pos > 0 else 1.0

    clf = XGBClassifier(
        objective="binary:logistic",
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        gamma=gamma,
        min_child_weight=min_child_weight,
        random_state=random_state,
        n_jobs=n_jobs,
        tree_method="hist",  # use "gpu_hist" if you have GPU support
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
    )

    clf.fit(
        x_train,
        y_train,
        eval_set=[(x_test, y_test)],
        verbose=verbose,
    )

    y_pred = clf.predict(x_test)
    y_prob_test = clf.predict_proba(x_test)[:, 1]
    y_prob_train = clf.predict_proba(x_train)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # Specificity
    labels = np.unique(y_test[~pd.isna(y_test)])
    if set(labels.tolist()) == {0, 1}:
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    else:
        specificity = np.nan

    # AUROC / AUPRC
    fpr, tpr, _ = roc_curve(y_test, y_prob_test)
    roc_auc = auc(fpr, tpr)

    prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_prob_test)
    pr_auc = auc(rec_arr, prec_arr)

    metrics = {
        "Accuracy": float(accuracy),
        "Balanced Accuracy": float(balanced_acc),
        "AUROC": float(roc_auc),
        "AUPRC": float(pr_auc),
        "MCC": float(mcc),
        "Precision": float(precision),
        "Sensitivity": float(recall),
        "Specificity": float(specificity),
        "F1": float(f1),
    }

    return TrainResult(
        metrics=metrics,
        y_prob_train=y_prob_train,
        y_prob_test=y_prob_test,
        y_train_true=y_train,
        y_test_true=y_test,
    )


def load_split(prefix: str, fp_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load train/test X and y arrays for a given fingerprint name."""
    fp_file = fp_name.lower()

    x_train_path = f"{prefix}_x_train_{fp_file}.csv"
    x_test_path = f"{prefix}_x_test_{fp_file}.csv"
    y_train_path = f"{prefix}_y_train.csv"
    y_test_path = f"{prefix}_y_test.csv"

    x_train = pd.read_csv(x_train_path, index_col=0).values
    x_test = pd.read_csv(x_test_path, index_col=0).values
    y_train = pd.read_csv(y_train_path, index_col=0).values.ravel()
    y_test = pd.read_csv(y_test_path, index_col=0).values.ravel()
    return x_train, x_test, y_train, y_test


def ensure_output_dir() -> str:
    """Create a timestamped output folder and return its path."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join("Prob_Inflampred_external", f"Prob_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nOutput folder for probabilities: {out_dir}")
    return out_dir


def run_all_fingerprints(fingerprints: List[str], *, num_runs: int = 3) -> Dict[str, Dict[str, Tuple[float, float]]]:
    results_by_fp: Dict[str, Dict[str, Tuple[float, float]]] = {}
    all_metrics_rows: List[Dict[str, float]] = []

    prob_dir = ensure_output_dir()

    for fp in fingerprints:
        fp_upper = fp.upper()
        print(f"\n=== Evaluating fingerprint: {fp_upper} ===")

        try:
            x_train, x_test, y_train, y_test = load_split(BASE_PREFIX, fp)
        except FileNotFoundError as e:
            print(f"[SKIP] Missing file(s) for {fp_upper}: {e}")
            continue

        per_metric_values: Dict[str, List[float]] = {k: [] for k in METRIC_KEYS}

        for run_idx in range(num_runs):
            seed = 42 + run_idx
            result = train_xgboost(
                x_train,
                x_test,
                y_train,
                y_test,
                n_estimators=500,
                max_depth=6,
                random_state=seed,
                n_jobs=-1,
                verbose=True,
            )

            # Record metrics
            row = dict(result.metrics)
            row["Fingerprint"] = fp_upper
            row["Run"] = run_idx + 1
            row["Seed"] = seed
            all_metrics_rows.append(row)

            for k in METRIC_KEYS:
                per_metric_values[k].append(result.metrics[k])

            # Save per-run probabilities
            train_df = pd.DataFrame({"y_true": result.y_train_true, "y_prob": result.y_prob_train})
            test_df = pd.DataFrame({"y_true": result.y_test_true, "y_prob": result.y_prob_test})

            train_path = os.path.join(prob_dir, f"{BASE_PREFIX}_train_prob_{fp.lower()}_run{run_idx+1}.csv")
            test_path = os.path.join(prob_dir, f"{BASE_PREFIX}_test_prob_{fp.lower()}_run{run_idx+1}.csv")

            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)
            print(f"Saved: {train_path} | {test_path}")

        # Mean ± SD across runs
        summary = {k: (float(np.nanmean(v)), float(np.nanstd(v))) for k, v in per_metric_values.items()}
        results_by_fp[fp_upper] = summary

        print(f"\n--- {fp_upper} Results (Mean ± SD over {num_runs} runs) ---")
        for k, (mean_val, std_val) in summary.items():
            print(f"{k}: {mean_val:.2f} ± {std_val:.2f}")

    # Save raw per-run metrics
    raw_path = f"{BASE_PREFIX}_XGB_fingerprint_metrics_raw.csv"
    pd.DataFrame(all_metrics_rows).to_csv(raw_path, index=False)
    print(f"\nSaved raw results: {raw_path}")

    return results_by_fp


def export_summary(results_by_fp: Dict[str, Dict[str, Tuple[float, float]]]) -> str:
    """Export Mean±SD summary table to CSV and return the output path."""
    df = pd.DataFrame(
        {
            fp: {metric: f"{mean:.2f} ± {std:.2f}" for metric, (mean, std) in metrics.items()}
            for fp, metrics in results_by_fp.items()
        }
    ).T
    out_path = f"{BASE_PREFIX}_XGB_fingerprint_metrics.csv"
    df.to_csv(out_path, index=True)
    return out_path


def main() -> None:
    fingerprints = ["ecfp", "maccs", "rdkit"]
    results_by_fp = run_all_fingerprints(fingerprints, num_runs=3)
    summary_path = export_summary(results_by_fp)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
