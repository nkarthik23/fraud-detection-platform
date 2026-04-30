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

# Controlled data-quality issue probabilities
P_MALFORMED_LINE = 0.01
P_DUPLICATE_TXN_ID = 0.01
P_INCONSISTENT_AMOUNT_FORMAT = 0.02
P_BAD_TIMESTAMP = 0.01
P_MISSING_OPTIONAL_FIELDS = 0.02
P_CASING_WHITESPACE_ISSUES = 0.02
P_NULL_BALANCE_FIELD = 0.02


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


def format_amount(amount: float, rng: random.Random) -> str:
    """
    Emit multiple money formats to simulate heterogeneous upstream systems.
    """
    if rng.random() >= P_INCONSISTENT_AMOUNT_FORMAT:
        return f"${amount:.2f}"

    style = rng.choice(["comma_dollar", "usd_prefix", "plain"])
    if style == "comma_dollar":
        return f"${amount:,.2f}"
    if style == "usd_prefix":
        return f"USD {amount:.2f}"
    return f"{amount:.2f}"


def maybe_bad_timestamp(ts: str, rng: random.Random) -> str:
    """
    Inject occasional invalid timestamp strings.
    """
    if rng.random() >= P_BAD_TIMESTAMP:
        return ts
    return rng.choice(["BAD_TS", "2026-99-99T99:99:99Z", "2026/01/01 10:00:00", ""])


def maybe_casing_whitespace(txn_type: str, merchant: str, rng: random.Random) -> tuple[str, str]:
    """
    Add casing and whitespace inconsistencies.
    """
    if rng.random() >= P_CASING_WHITESPACE_ISSUES:
        return txn_type, merchant

    type_variant = rng.choice([txn_type.lower(), txn_type.title(), f"  {txn_type}  "])
    merchant_variant = rng.choice([merchant.lower(), merchant.title(), f"  {merchant}  "])
    return type_variant, merchant_variant


def maybe_missing_optional_fields(merchant: str, txn_type: str, rng: random.Random) -> tuple[str, str]:
    """
    Drop merchant/type occasionally to simulate incomplete source records.
    """
    if rng.random() >= P_MISSING_OPTIONAL_FIELDS:
        return merchant, txn_type

    if rng.random() < 0.5:
        merchant = ""
    else:
        txn_type = ""
    return merchant, txn_type


def maybe_null_balance(value: float, rng: random.Random) -> str:
    """
    Emit empty/null-like balance fields.
    """
    if rng.random() >= P_NULL_BALANCE_FIELD:
        return f"{value:.2f}"
    return rng.choice(["", "null", "NA"])


def maybe_corrupt_line(line: str, rng: random.Random) -> str:
    """
    Produce malformed lines that parser should skip.
    """
    if rng.random() >= P_MALFORMED_LINE:
        return line
    return rng.choice(
        [
            line.replace("|", " ", 3),  # break delimiters
            line.replace("txn=", "txnid=", 1),  # rename required key
            f"MALFORMED::{line[: max(0, len(line) // 2)]}",
            "this is not a transaction line",
        ]
    )


def build_log_line(
    row: pd.Series,
    idx: int,
    base_time: datetime,
    rng: random.Random,
    duplicate_txn_id: str | None = None,
) -> str:
    """
    Build one messy semi-structured raw log line.
    """
    timestamp = step_to_timestamp(row["step"], base_time)
    timestamp = maybe_bad_timestamp(timestamp, rng)
    txn_id = duplicate_txn_id or to_transaction_id(row["nameOrig"], row["nameDest"], row["step"], idx)
    user_id = to_user_id(row["nameOrig"])
    amount = float(row["amount"])
    txn_type = str(row["type"]).strip().upper()
    merchant = txn_type  # requested format mirrors type in merchant field

    amount_str = format_amount(amount, rng)
    txn_type, merchant = maybe_casing_whitespace(txn_type, merchant, rng)
    merchant, txn_type = maybe_missing_optional_fields(merchant, txn_type, rng)

    old_org = float(row["oldbalanceOrg"])
    new_org = float(row["newbalanceOrig"])
    old_dest = float(row["oldbalanceDest"])
    new_dest = float(row["newbalanceDest"])
    is_fraud = int(row["isFraud"])

    old_org_str = maybe_null_balance(old_org, rng)
    new_org_str = maybe_null_balance(new_org, rng)
    old_dest_str = maybe_null_balance(old_dest, rng)
    new_dest_str = maybe_null_balance(new_dest, rng)

    line = (
        f"{timestamp} | txn={txn_id} | user={user_id} | amt=${amount:.2f} | "
        f"merchant=\"{merchant}\" | type={txn_type} | "
        f"oldbalanceOrg={old_org_str} | newbalanceOrig={new_org_str} | "
        f"oldbalanceDest={old_dest_str} | newbalanceDest={new_dest_str} | "
        f"isFraud={is_fraud}"
    )
    line = line.replace(f"amt=${amount:.2f}", f"amt={amount_str}")
    return maybe_corrupt_line(line, rng)


def main() -> None:
    if not PAYSIM_PATH.exists():
        raise FileNotFoundError(f"PaySim CSV not found: {PAYSIM_PATH}")

    random.seed(RANDOM_SEED)
    rng = random.Random(RANDOM_SEED)

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

    lines: list[str] = []
    prior_txn_ids: list[str] = []
    duplicate_rows = 0

    for idx, row in df.iterrows():
        duplicate_txn_id = None
        if prior_txn_ids and rng.random() < P_DUPLICATE_TXN_ID:
            duplicate_txn_id = rng.choice(prior_txn_ids)
            duplicate_rows += 1
        else:
            new_txn_id = to_transaction_id(row["nameOrig"], row["nameDest"], row["step"], idx)
            prior_txn_ids.append(new_txn_id)

        lines.append(build_log_line(row, idx, base_time, rng, duplicate_txn_id=duplicate_txn_id))

    OUTPUT_LOG_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote raw log lines: {len(lines):,}")
    print(f"Injected duplicate transaction IDs: {duplicate_rows:,}")
    print(f"Output path: {OUTPUT_LOG_PATH}")


if __name__ == "__main__":
    main()