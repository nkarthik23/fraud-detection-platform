from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
PAYSIM_PATH = RAW_DIR / "PS_20174392719_1491204439457_log.csv"
OUTPUT_LOG_PATH = RAW_DIR / "transactions.log"

MAX_ROWS = 50_000
RANDOM_SEED = 42


def step_to_timestamp(step: int, base: datetime) -> str:
    """
    Convert PaySim step (hour index) to ISO UTC timestamp string.
    """
    ts = base + timedelta(hours=int(step))
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def to_user_id(name_orig: str) -> str:
    """
    Convert PaySim account id to normalized user id format.
    Example: C1231006815 -> U1231006815
    """
    digits = "".join(ch for ch in str(name_orig) if ch.isdigit())
    return f"U{digits}" if digits else "U0"


def to_transaction_id(name_orig: str, name_dest: str, step: int, idx: int) -> str:
    """
    Build deterministic-ish transaction id from source fields.
    """
    return f"{name_orig}_{name_dest}_{step}_{idx}"


def build_log_line(row: pd.Series, idx: int, base_time: datetime) -> str:
    """
    Build one messy semi-structured raw log line.
    """
    timestamp = step_to_timestamp(row["step"], base_time)
    txn_id = to_transaction_id(row["nameOrig"], row["nameDest"], row["step"], idx)
    user_id = to_user_id(row["nameOrig"])
    amount = float(row["amount"])
    txn_type = str(row["type"]).strip().upper()
    merchant = txn_type  # requested format mirrors type in merchant field
    old_org = float(row["oldbalanceOrg"])
    new_org = float(row["newbalanceOrig"])
    old_dest = float(row["oldbalanceDest"])
    new_dest = float(row["newbalanceDest"])
    is_fraud = int(row["isFraud"])

    return (
        f"{timestamp} | txn={txn_id} | user={user_id} | amt=${amount:.2f} | "
        f"merchant=\"{merchant}\" | type={txn_type} | "
        f"oldbalanceOrg={old_org:.2f} | newbalanceOrig={new_org:.2f} | "
        f"oldbalanceDest={old_dest:.2f} | newbalanceDest={new_dest:.2f} | "
        f"isFraud={is_fraud}"
    )


def main() -> None:
    if not PAYSIM_PATH.exists():
        raise FileNotFoundError(f"PaySim CSV not found: {PAYSIM_PATH}")

    random.seed(RANDOM_SEED)

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    use_cols = [
        "step",
        "type",
        "amount",
        "nameOrig",
        "oldbalanceOrg",
        "newbalanceOrig",
        "nameDest",
        "oldbalanceDest",
        "newbalanceDest",
        "isFraud",
    ]

    df = pd.read_csv(PAYSIM_PATH, usecols=use_cols, nrows=MAX_ROWS)
    print(f"Read PaySim rows: {len(df):,}")

    # Shuffle to avoid only early-step patterns
    df = df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    base_time = datetime(2026, 1, 1, 0, 0, 0)

    lines = [build_log_line(row, idx, base_time) for idx, row in df.iterrows()]

    OUTPUT_LOG_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote raw log lines: {len(lines):,}")
    print(f"Output path: {OUTPUT_LOG_PATH}")


if __name__ == "__main__":
    main()