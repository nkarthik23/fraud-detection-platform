from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

INPUT_LOG_PATH = Path("data/raw/transactions.log")
OUTPUT_PARQUET_PATH = Path("data/processed/transactions_clean.parquet")

LINE_PATTERN = re.compile(
    r'^(?P<timestamp>\S+)\s+\|\s+'
    r'txn=(?P<transaction_id>[^|]+?)\s+\|\s+'
    r'user=(?P<user_id>[^|]+?)\s+\|\s+'
    r'amt=\$(?P<amount>[\d\.]+)\s+\|\s+'
    r'merchant="(?P<merchant>[^"]+)"\s+\|\s+'
    r'type=(?P<transaction_type>[^|]+?)\s+\|\s+'
    r'oldbalanceOrg=(?P<oldbalanceOrg>[\d\.]+)\s+\|\s+'
    r'newbalanceOrig=(?P<newbalanceOrig>[\d\.]+)\s+\|\s+'
    r'oldbalanceDest=(?P<oldbalanceDest>[\d\.]+)\s+\|\s+'
    r'newbalanceDest=(?P<newbalanceDest>[\d\.]+)\s+\|\s+'
    r'isFraud=(?P<is_fraud>[01])$'
)


def parse_line(line: str) -> dict | None:
    match = LINE_PATTERN.match(line.strip())
    if not match:
        return None
    return match.groupdict()


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

    # Type conversions
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["oldbalanceOrg"] = pd.to_numeric(df["oldbalanceOrg"], errors="coerce")
    df["newbalanceOrig"] = pd.to_numeric(df["newbalanceOrig"], errors="coerce")
    df["oldbalanceDest"] = pd.to_numeric(df["oldbalanceDest"], errors="coerce")
    df["newbalanceDest"] = pd.to_numeric(df["newbalanceDest"], errors="coerce")
    df["is_fraud"] = pd.to_numeric(df["is_fraud"], errors="coerce").astype("Int64")

    # Standardize transaction type
    df["transaction_type"] = df["transaction_type"].str.upper().str.strip()

    # Drop rows with failed type conversions
    before_type_drop = len(df)
    df = df.dropna(
        subset=[
            "timestamp",
            "amount",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
            "is_fraud",
        ]
    )
    conversion_dropped = before_type_drop - len(df)

    # Deduplicate by transaction_id
    before_dedupe = len(df)
    df = df.drop_duplicates(subset=["transaction_id"], keep="first")
    duplicates_removed = before_dedupe - len(df)

    # Force final dtypes
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
    print(f"Rows dropped after type conversion: {conversion_dropped:,}")
    print(f"Duplicates removed: {duplicates_removed:,}")
    print(f"Final clean rows: {len(df):,}")
    print(f"Output path: {OUTPUT_PARQUET_PATH}")


if __name__ == "__main__":
    main()