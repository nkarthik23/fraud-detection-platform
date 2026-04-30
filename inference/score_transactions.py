from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd


MODEL_PATH = Path("models/best_model.pkl")
FEATURES_PATH = Path("data/processed/training_features.parquet")
OUTPUT_DIR = Path("data/scored")
OUTPUT_PATH = OUTPUT_DIR / "scored_transactions.parquet"

MODEL_VERSION = "best_model.pkl"
LABEL_CANDIDATES = ["is_fraud", "isFraud", "label", "target"]


def load_model(model_path: Path = MODEL_PATH):
    """Load trained model artifact from disk."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    print(f"[LOAD] Loading model from {model_path}")
    return joblib.load(model_path)


def load_features(features_path: Path = FEATURES_PATH) -> pd.DataFrame:
    """Load feature dataset to be scored."""
    if not features_path.exists():
        raise FileNotFoundError(f"Feature dataset not found: {features_path}")
    print(f"[LOAD] Loading features from {features_path}")
    df = pd.read_parquet(features_path)
    print(f"[LOAD] Rows loaded: {len(df):,}")
    return df


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series | None, str | None]:
    """
    Split input into model matrix X and optional label vector y.
    Drops common label columns if present.
    """
    label_col = next((col for col in LABEL_CANDIDATES if col in df.columns), None)
    y = df[label_col].astype(int) if label_col is not None else None
    X = df.drop(columns=[label_col]) if label_col is not None else df.copy()

    # Match training-time preprocessing: convert timestamp to epoch seconds if present.
    if "timestamp" in X.columns:
        ts = pd.to_datetime(X["timestamp"], errors="coerce", utc=True)
        X["timestamp"] = ts.astype("int64") // 10**9

    print(f"[PREP] Label column: {label_col if label_col else 'not present'}")
    print(f"[PREP] Feature columns used for scoring: {X.shape[1]}")
    return X, y, label_col


def assign_decision(probabilities: np.ndarray) -> np.ndarray:
    """
    Apply business decision policy:
      >=0.90 -> BLOCK
      0.70-0.90 -> REVIEW
      <0.70 -> ALLOW
    """
    decisions = np.where(
        probabilities >= 0.90,
        "BLOCK",
        np.where(probabilities >= 0.70, "REVIEW", "ALLOW"),
    )
    return decisions


def score_transactions(model, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Generate fraud probabilities and mapped decisions."""
    print("[SCORE] Running model.predict_proba(...)")
    probabilities = model.predict_proba(X)[:, 1]
    decisions = assign_decision(probabilities)
    return probabilities, decisions


def evaluate_decisions(decisions: np.ndarray, y: pd.Series | None) -> None:
    """Print decision summary and label-aware quality summary when y is available."""
    total = len(decisions)
    allow_count = int(np.sum(decisions == "ALLOW"))
    review_count = int(np.sum(decisions == "REVIEW"))
    block_count = int(np.sum(decisions == "BLOCK"))

    print("\n[SUMMARY] Decision breakdown")
    print(f"  Total scored: {total:,}")
    print(f"  ALLOW:        {allow_count:,}")
    print(f"  REVIEW:       {review_count:,}")
    print(f"  BLOCK:        {block_count:,}")

    if y is None:
        return

    fraud_mask = y.to_numpy() == 1
    nonfraud_mask = ~fraud_mask

    frauds_blocked = int(np.sum((decisions == "BLOCK") & fraud_mask))
    frauds_review = int(np.sum((decisions == "REVIEW") & fraud_mask))
    frauds_missed = int(np.sum((decisions == "ALLOW") & fraud_mask))

    false_positive_blocks = int(np.sum((decisions == "BLOCK") & nonfraud_mask))
    false_positive_reviews = int(np.sum((decisions == "REVIEW") & nonfraud_mask))

    print("\n[SUMMARY] Label-aware policy outcomes")
    print(f"  Frauds blocked:          {frauds_blocked:,}")
    print(f"  Frauds sent to review:   {frauds_review:,}")
    print(f"  Frauds allowed/missed:   {frauds_missed:,}")
    print(f"  False positive blocks:   {false_positive_blocks:,}")
    print(f"  False positive reviews:  {false_positive_reviews:,}")


def save_scored_transactions(df: pd.DataFrame, output_path: Path = OUTPUT_PATH) -> None:
    """Persist scored transactions to parquet."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"\n[SAVE] Scored output written to {output_path}")


def main() -> None:
    model = load_model()
    raw_df = load_features()
    X, y, _ = prepare_features(raw_df)

    probabilities, decisions = score_transactions(model, X)

    scored_df = raw_df.copy()
    scored_df["fraud_probability"] = probabilities
    scored_df["decision"] = decisions
    scored_df["model_version"] = MODEL_VERSION
    scored_df["scored_at"] = datetime.now(timezone.utc).isoformat()

    evaluate_decisions(decisions, y)
    save_scored_transactions(scored_df)


if __name__ == "__main__":
    main()
