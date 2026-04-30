from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

INPUT_LOG_PATH = Path("data/raw/transactions.log")
OUTPUT_PARQUET_PATH = Path("data/processed/transactions_clean.parquet")

AMOUNT_PATTERN = re.compile(r"^\$?\s*([\d,]+(?:\.\d+)?)$|^USD\s+([\d,]+(?:\.\d+)?)$", re.IGNORECASE)
BALANCE_NULL_TOKENS = {"", "null", "na", "none"}


def parse_line(line: str) -> dict | None:
    parts = [part.strip() for part in line.split("|")]
    if len(parts) < 11:
        return None

    timestamp = parts[0]
    kv: dict[str, str] = {}
    for part in parts[1:]:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        kv[key.strip()] = value.strip()

    required_keys = {
        "txn",
        "user",
        "amt",
        "merchant",
        "type",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "isFraud",
    }
    if not required_keys.issubset(kv.keys()):
        return None

    merchant = kv["merchant"].strip().strip('"')
    transaction_type = kv["type"].strip()
    if not merchant:
        merchant = "UNKNOWN"
    if not transaction_type:
        transaction_type = "UNKNOWN"

    return {
        "timestamp": timestamp,
        "transaction_id": kv["txn"].strip(),
        "user_id": kv["user"].strip(),
        "amount": kv["amt"].strip(),
        "merchant": merchant,
        "transaction_type": transaction_type,
        "oldbalanceOrg": kv["oldbalanceOrg"].strip(),
        "newbalanceOrig": kv["newbalanceOrig"].strip(),
        "oldbalanceDest": kv["oldbalanceDest"].strip(),
        "newbalanceDest": kv["newbalanceDest"].strip(),
        "is_fraud": kv["isFraud"].strip(),
    }


def parse_amount(value: str) -> float | None:
    raw = value.strip()
    match = AMOUNT_PATTERN.match(raw)
    if not match:
        return None
    numeric = match.group(1) or match.group(2)
    if numeric is None:
        return None
    try:
        return float(numeric.replace(",", ""))
    except ValueError:
        return None


def parse_balance(value: str) -> float | None:
    raw = value.strip().lower()
    if raw in BALANCE_NULL_TOKENS:
        return None
    try:
        return float(raw.replace(",", ""))
    except ValueError:
        return None


def main() -> None:
    if not INPUT_LOG_PATH.exists():
        raise FileNotFoundError(f"Input log not found: {INPUT_LOG_PATH}")

    OUTPUT_PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)

    lines = INPUT_LOG_PATH.read_text(encoding="utf-8").splitlines()
    total_rows = len(lines)

    parsed_records: list[dict] = []
    malformed_rows = 0

    for line in lines:
        parsed = parse_line(line)
        if parsed is None:
            malformed_rows += 1
            continue
        parsed_records.append(parsed)

    if not parsed_records:
        raise ValueError("No valid rows parsed from transactions.log")

    df = pd.DataFrame(parsed_records)

    # Robust type conversions
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["amount"] = df["amount"].apply(parse_amount)
    df["oldbalanceOrg"] = df["oldbalanceOrg"].apply(parse_balance)
    df["newbalanceOrig"] = df["newbalanceOrig"].apply(parse_balance)
    df["oldbalanceDest"] = df["oldbalanceDest"].apply(parse_balance)
    df["newbalanceDest"] = df["newbalanceDest"].apply(parse_balance)
    df["is_fraud"] = pd.to_numeric(df["is_fraud"], errors="coerce")

    # Standardize transaction type
    df["transaction_type"] = df["transaction_type"].str.upper().str.strip()

    # Drop rows with invalid timestamps
    before_ts_drop = len(df)
    df = df.dropna(subset=["timestamp"])
    invalid_timestamps_dropped = before_ts_drop - len(df)

    # Drop rows with invalid required numeric fields
    before_numeric_drop = len(df)
    df = df.dropna(
        subset=["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "is_fraud"]
    )
    invalid_numeric_rows_dropped = before_numeric_drop - len(df)

    # Deduplicate by transaction_id
    before_dedupe = len(df)
    df = df.drop_duplicates(subset=["transaction_id"], keep="first")
    duplicates_removed = before_dedupe - len(df)

    # Force final dtypes
    df["amount"] = df["amount"].astype(float)
    df["oldbalanceOrg"] = df["oldbalanceOrg"].astype(float)
    df["newbalanceOrig"] = df["newbalanceOrig"].astype(float)
    df["oldbalanceDest"] = df["oldbalanceDest"].astype(float)
    df["newbalanceDest"] = df["newbalanceDest"].astype(float)
    df["is_fraud"] = df["is_fraud"].astype(int)

    # Schema validation (required columns)
    required_cols = [
        "timestamp",
        "transaction_id",
        "user_id",
        "amount",
        "merchant",
        "transaction_type",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "is_fraud",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after parsing: {missing}")

    df = df[required_cols]

    df.to_parquet(OUTPUT_PARQUET_PATH, index=False, engine="pyarrow")

    print(f"Rows read: {total_rows:,}")
    print(f"Malformed rows skipped: {malformed_rows:,}")
    print(f"Invalid timestamps dropped: {invalid_timestamps_dropped:,}")
    print(f"Invalid numeric rows dropped: {invalid_numeric_rows_dropped:,}")
    print(f"Duplicates removed: {duplicates_removed:,}")
    print(f"Final clean rows: {len(df):,}")
    print(f"Output path: {OUTPUT_PARQUET_PATH}")


if __name__ == "__main__":
    main()