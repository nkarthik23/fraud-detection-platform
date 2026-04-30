from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


# Project paths (relative only)
DATA_PATH = Path("data/processed/training_features.parquet")
MODELS_DIR = Path("models")
MODEL_OUTPUT_PATH = MODELS_DIR / "best_model.pkl"
METRICS_OUTPUT_PATH = MODELS_DIR / "metrics.json"
LEAKAGE_CHECK_DROP_COLUMNS = ["origin_balance_error", "dest_balance_error"]


def evaluate_binary_classifier(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    """
    Compute fraud-relevant metrics for imbalanced binary classification.
    We intentionally skip accuracy because it's misleading for rare fraud cases.
    """
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),  # [[tn, fp], [fn, tp]]
    }

def print_metrics(model_name: str, metrics: dict) -> None:
    """
    Pretty-print model metrics in a readable way.
    """
    print(f"\n=== {model_name} ===")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1        : {metrics['f1']:.4f}")
    print(f"ROC-AUC   : {metrics['roc_auc']:.4f}")
    print(f"PR-AUC    : {metrics['pr_auc']:.4f}")
    print(f"Confusion : {metrics['confusion_matrix']}")


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build preprocessing pipeline:
    - Numeric columns: impute missing values + standardize
    - Categorical columns: impute missing values + one-hot encode
    """
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )


def main() -> None:
    # Ensure output directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # -----------------------------
    # 1) Load feature dataset
    # -----------------------------
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Input dataset not found: {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    if "is_fraud" not in df.columns:
        raise ValueError("Target column 'is_fraud' not found in dataset.")
    # If timestamp exists, convert to numeric epoch seconds for model compatibility.
    # (If your feature file already has hour-based features and no timestamp, this safely does nothing.)
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df["timestamp"] = ts.astype("int64") // 10**9  # seconds since epoch
    # Separate features and target
    y = df["is_fraud"].astype(int)
    X = df.drop(columns=["is_fraud"])
    # Sanity check: temporarily drop potentially leakage-prone features.
    # This helps verify whether model quality depends on near-deterministic columns.
    drop_cols = [col for col in LEAKAGE_CHECK_DROP_COLUMNS if col in X.columns]
    if drop_cols:
        X = X.drop(columns=drop_cols)
        print(f"Sanity check enabled: dropped columns {drop_cols}")
    # Print class balance (important for fraud modeling)
    fraud_count = int(y.sum())
    total_count = int(len(y))
    fraud_ratio = fraud_count / total_count if total_count else 0.0
    print(f"Rows loaded: {total_count:,}")
    print(f"Fraud class balance: {fraud_count:,}/{total_count:,} ({fraud_ratio:.2%})")
    # -----------------------------
    # 2) Stratified train/test split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,  # critical for imbalanced labels
    )
    # Build shared preprocessor once from training schema
    preprocessor = build_preprocessor(X_train)
    # Class imbalance handling ratio for weighted models
    negatives = int((y_train == 0).sum())
    positives = int((y_train == 1).sum())
    scale_pos_weight = (negatives / positives) if positives > 0 else 1.0
    # -----------------------------
    # 3) Logistic Regression baseline
    # -----------------------------
    logistic_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",  # helps in imbalanced fraud data
                    random_state=42,
                ),
            ),
        ]
    )
    logistic_pipeline.fit(X_train, y_train)
    lr_pred = logistic_pipeline.predict(X_test)
    lr_proba = logistic_pipeline.predict_proba(X_test)[:, 1]
    lr_metrics = evaluate_binary_classifier(y_test.to_numpy(), lr_pred, lr_proba)
    print_metrics("Logistic Regression", lr_metrics)
    # -----------------------------
    # 4) XGBoost stronger baseline
    # -----------------------------
    xgb_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                XGBClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="binary:logistic",
                    eval_metric="aucpr",  # PR-oriented metric for imbalance
                    random_state=42,
                    scale_pos_weight=scale_pos_weight,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    xgb_pipeline.fit(X_train, y_train)
    xgb_pred = xgb_pipeline.predict(X_test)
    xgb_proba = xgb_pipeline.predict_proba(X_test)[:, 1]
    xgb_metrics = evaluate_binary_classifier(y_test.to_numpy(), xgb_pred, xgb_proba)
    print_metrics("XGBoost", xgb_metrics)
    # -----------------------------
    # 5) Pick better model
    # -----------------------------
    # For fraud, PR-AUC is often the best single selection metric.
    if xgb_metrics["pr_auc"] >= lr_metrics["pr_auc"]:
        best_model_name = "xgboost"
        best_model = xgb_pipeline
        best_metrics = xgb_metrics
    else:
        best_model_name = "logistic_regression"
        best_model = logistic_pipeline
        best_metrics = lr_metrics
    # Save best model
    joblib.dump(best_model, MODEL_OUTPUT_PATH)
    # Save summary metrics JSON
    metrics_payload = {
        "best_model": best_model_name,
        "selection_metric": "pr_auc",
        "logistic_regression": lr_metrics,
        "xgboost": xgb_metrics,
    }
    with METRICS_OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print("\n=== Artifacts Saved ===")
    print(f"Best model: {best_model_name}")
    print(f"Model path: {MODEL_OUTPUT_PATH}")
    print(f"Metrics path: {METRICS_OUTPUT_PATH}")
if __name__ == "__main__":
    main()
