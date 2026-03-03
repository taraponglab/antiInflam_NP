"""
BiLSTM training and evaluation across multiple fingerprint feature sets.

Expected input files:
    {BASE_PREFIX}_x_train_<fp>.csv
    {BASE_PREFIX}_x_test_<fp>.csv
    {BASE_PREFIX}_y_train.csv
    {BASE_PREFIX}_y_test.csv

Outputs:
    - Per-run predicted probabilities (train/test) saved under:
        Prob_<BASE_PREFIX>/Prob_<timestamp>/
    - Raw per-run metrics:
        {BASE_PREFIX}_BiLSTM_fingerprint_metrics_raw.csv
    - Mean ± SD summary:
        {BASE_PREFIX}_BiLSTM_fingerprint_metrics.csv
"""

from __future__ import annotations

import os
import random
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
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
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


# ===== USER CONFIGURATION =====
BASE_PREFIX = "InFlam2_full"


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


# ===== MODEL ARCHITECTURE =====
def build_bilstm_model(input_dim: int) -> tf.keras.Model:
    """Build and compile a BiLSTM binary classifier."""
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=(1, input_dim)),
        Dropout(0.3),
        Bidirectional(LSTM(32)),
        Dropout(0.3),
        Dense(100, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ===== TRAIN + EVALUATE =====
def evaluate_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    epochs: int = 30,
    batch_size: int = 32,
    seed: int = 42,
    verbose: int = 1,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Train BiLSTM model and compute evaluation metrics."""

    # Reproducibility
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Reshape for LSTM input: (samples, timesteps=1, features)
    x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
    x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

    model = build_bilstm_model(x_train.shape[2])

    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=verbose,
    )

    # Predict probabilities
    y_test_prob = model.predict(x_test, verbose=0).ravel()
    y_train_prob = model.predict(x_train, verbose=0).ravel()

    y_test_pred = (y_test_prob > 0.5).astype(int)

    # Classification metrics
    acc = accuracy_score(y_test, y_test_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_test_pred)
    mcc = matthews_corrcoef(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, zero_division=0)
    recall = recall_score(y_test, y_test_pred, zero_division=0)
    f1 = f1_score(y_test, y_test_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    # AUROC
    fpr, tpr, _ = roc_curve(y_test, y_test_prob)
    roc_auc = auc(fpr, tpr)

    # AUPRC
    prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_test_prob)
    pr_auc = auc(rec_arr, prec_arr)

    metrics = {
        "Accuracy": float(acc),
        "Balanced Accuracy": float(balanced_acc),
        "AUROC": float(roc_auc),
        "AUPRC": float(pr_auc),
        "MCC": float(mcc),
        "Precision": float(precision),
        "Sensitivity": float(recall),
        "Specificity": float(specificity),
        "F1": float(f1),
    }

    return metrics, y_train_prob, y_test_prob, y_train, y_test


# ===== DATA LOADING =====
def load_split(prefix: str, fingerprint: str):
    fp = fingerprint.lower()

    x_train = pd.read_csv(f"{prefix}_x_train_{fp}.csv", index_col=0).values
    x_test = pd.read_csv(f"{prefix}_x_test_{fp}.csv", index_col=0).values
    y_train = pd.read_csv(f"{prefix}_y_train.csv", index_col=0).values.ravel()
    y_test = pd.read_csv(f"{prefix}_y_test.csv", index_col=0).values.ravel()

    return x_train, x_test, y_train, y_test


def create_output_folder() -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder = os.path.join(f"Prob_{BASE_PREFIX}", f"Prob_{timestamp}")
    os.makedirs(folder, exist_ok=True)
    print(f"\nProbability output folder: {folder}")
    return folder


# ===== RUN ALL FINGERPRINTS =====
def run_all_fingerprints(fingerprints: List[str], num_runs: int = 3):
    results_by_fp = {}
    all_metrics_rows = []
    prob_folder = create_output_folder()

    for fp in fingerprints:
        print(f"\n=== Evaluating fingerprint: {fp.upper()} ===")

        try:
            x_train, x_test, y_train, y_test = load_split(BASE_PREFIX, fp)
        except FileNotFoundError as e:
            print(f"[SKIP] Missing file(s) for {fp.upper()}: {e}")
            continue

        metric_storage = {k: [] for k in METRIC_KEYS}

        for run in range(num_runs):
            seed = 42 + run
            print(f"Run {run+1}/{num_runs} (seed={seed})")

            metrics, y_train_prob, y_test_prob, y_train_true, y_test_true = evaluate_model(
                x_train,
                y_train,
                x_test,
                y_test,
                epochs=30,
                batch_size=32,
                seed=seed,
                verbose=1,
            )

            for k in METRIC_KEYS:
                metric_storage[k].append(metrics[k])

            metrics.update({
                "Fingerprint": fp.upper(),
                "Run": run + 1,
                "Seed": seed,
            })
            all_metrics_rows.append(metrics)

            # Save probabilities
            pd.DataFrame({"y_true": y_train_true, "y_prob": y_train_prob}).to_csv(
                os.path.join(prob_folder, f"{BASE_PREFIX}_train_prob_{fp}_run{run+1}.csv"),
                index=False,
            )

            pd.DataFrame({"y_true": y_test_true, "y_prob": y_test_prob}).to_csv(
                os.path.join(prob_folder, f"{BASE_PREFIX}_test_prob_{fp}_run{run+1}.csv"),
                index=False,
            )

        # Mean ± SD
        summary = {
            k: (float(np.nanmean(v)), float(np.nanstd(v)))
            for k, v in metric_storage.items()
        }
        results_by_fp[fp.upper()] = summary

        print(f"\n--- {fp.upper()} Results (Mean ± SD) ---")
        for k, (mean_val, std_val) in summary.items():
            print(f"{k}: {mean_val:.3f} ± {std_val:.3f}")

    # Save raw metrics
    raw_path = f"{BASE_PREFIX}_BiLSTM_fingerprint_metrics_raw.csv"
    pd.DataFrame(all_metrics_rows).to_csv(raw_path, index=False)
    print(f"\nSaved raw results: {raw_path}")

    return results_by_fp


def export_summary(results_by_fp: Dict[str, Dict[str, Tuple[float, float]]]):
    df = pd.DataFrame({
        fp: {metric: f"{mean:.3f} ± {std:.3f}" for metric, (mean, std) in metrics.items()}
        for fp, metrics in results_by_fp.items()
    }).T

    out_path = f"{BASE_PREFIX}_BiLSTM_fingerprint_metrics.csv"
    df.to_csv(out_path)
    print(f"Saved summary: {out_path}")


# ===== MAIN =====
def main():
    fingerprints = ["ecfp", "maccs", "rdkit", "phychem", "estate"]
    results = run_all_fingerprints(fingerprints, num_runs=3)
    export_summary(results)


if __name__ == "__main__":
    main()
