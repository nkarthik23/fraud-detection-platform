from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

INPUT_PARQUET_PATH = Path("data/processed/transactions_clean.parquet")
OUTPUT_PARQUET_PATH = Path("data/processed/training_features.parquet")


def main() -> None:
    if not INPUT_PARQUET_PATH.exists():
        raise FileNotFoundError(f"Input parquet not found: {INPUT_PARQUET_PATH}")

    OUTPUT_PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(INPUT_PARQUET_PATH, engine="pyarrow")

    # Ensure timestamp is datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])

    # Core features
    df["log_amount"] = np.log1p(df["amount"])
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["is_night_transaction"] = df["hour_of_day"].isin([0, 1, 2, 3, 4, 5, 23]).astype(int)

    df["balance_delta_origin"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["balance_delta_dest"] = df["newbalanceDest"] - df["oldbalanceDest"]

    df["origin_balance_error"] = (df["balance_delta_origin"] - df["amount"]).abs()
    df["dest_balance_error"] = (df["balance_delta_dest"] - df["amount"]).abs()

    # High amount threshold from data distribution
    high_amount_threshold = df["amount"].quantile(0.95)
    df["is_high_amount"] = (df["amount"] >= high_amount_threshold).astype(int)

    # Avoid division by zero
    df["amount_to_oldbalance_ratio"] = np.where(
        df["oldbalanceOrg"] > 0,
        df["amount"] / df["oldbalanceOrg"],
        0.0,
    )

    # One-hot encode transaction_type
    txn_dummies = pd.get_dummies(
        df["transaction_type"],
        prefix="transaction_type",
        dtype=np.int8,
    )

    # Final dataframe
    base_cols = [
        "transaction_id",
        "timestamp",
        "user_id",
        "amount",
        "merchant",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "log_amount",
        "hour_of_day",
        "is_night_transaction",
        "balance_delta_origin",
        "balance_delta_dest",
        "origin_balance_error",
        "dest_balance_error",
        "is_high_amount",
        "amount_to_oldbalance_ratio",
        "is_fraud",
    ]
    out_df = pd.concat([df[base_cols], txn_dummies], axis=1)

    out_df.to_parquet(OUTPUT_PARQUET_PATH, index=False, engine="pyarrow")

    fraud_rate = out_df["is_fraud"].mean()
    print(f"Row count: {len(out_df):,}")
    print(f"Fraud class balance: {out_df['is_fraud'].sum():,}/{len(out_df):,} ({fraud_rate:.2%})")
    print(f"Output path: {OUTPUT_PARQUET_PATH}")


if __name__ == "__main__":
    main()