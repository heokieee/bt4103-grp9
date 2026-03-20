
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

DEFAULT_ID_CANDIDATES = [
    "CustomerID",
    "customerID",
    "customer_id",
    "id",
    "ID",
    "firestore_doc_id",
]

def safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    return pd.read_csv(path) if path.exists() else None

def to_binary_series(series: pd.Series) -> pd.Series:
    s = series.copy()
    if s.dtype == object:
        mapping = {
            "Yes": 1, "No": 0,
            "yes": 1, "no": 0,
            "1": 1, "0": 0,
            1: 1, 0: 0,
            True: 1, False: 0,
        }
        s = s.map(mapping).fillna(s)
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)

def pick_customer_id_column(df: pd.DataFrame) -> Optional[str]:
    for col in DEFAULT_ID_CANDIDATES:
        if col in df.columns:
            return col
    for col in df.columns:
        col_l = col.lower()
        if "customer" in col_l and "id" in col_l:
            return col
    return None

def risk_tier_from_proba(p):
    p = float(p)
    if p < 0.25:
        return "Low"
    elif p < 0.50:
        return "Medium"
    elif p < 0.75:
        return "High"
    else:
        return "Critical"

def simple_action_recommendations(row: pd.Series) -> list[str]:
    actions = []

    if row.get("Complain", 0) == 1:
        actions.append("Follow up on recent complaint and resolve within 48 hours.")
    if row.get("SatisfactionScore", 5) <= 2:
        actions.append("Launch satisfaction recovery outreach with a personal touchpoint.")
    if row.get("CashbackAmount", 999) < 130:
        actions.append("Offer a cashback or retention incentive to improve stickiness.")
    if row.get("CouponUsed", 999) < 1:
        actions.append("Send a coupon campaign to encourage the next purchase.")
    if row.get("DaySinceLastOrder", 0) > 10:
        actions.append("Trigger a win-back campaign because the customer has been inactive.")
    if row.get("Tenure", 99) <= 2:
        actions.append("Prioritise onboarding and early lifecycle engagement.")

    if not actions:
        actions.append("No major immediate churn signals detected. Continue standard engagement.")

    return actions

def load_artifacts(
    pipeline_path: Path,
    rf_model_path: Path,
    xgb_model_path: Path,
    lgbm_model_path: Path,
    meta_path: Path,
):
    pipeline = joblib.load(pipeline_path)
    rf_model = joblib.load(rf_model_path)
    xgb_model = joblib.load(xgb_model_path)
    lgbm_model = joblib.load(lgbm_model_path)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return pipeline, rf_model, xgb_model, lgbm_model, meta

def preprocess_for_ensemble(
    df_input: pd.DataFrame,
    pipeline,
    selected_features: list[str],
) -> pd.DataFrame:
    X_processed = pipeline.transform(df_input)

    if isinstance(X_processed, pd.DataFrame):
        return X_processed

    X_processed = pd.DataFrame(X_processed)

    if len(selected_features) == X_processed.shape[1]:
        X_processed.columns = selected_features

    return X_processed

def predict_weighted_ensemble(
    X_processed: pd.DataFrame,
    rf_model,
    xgb_model,
    lgbm_model,
    w_rf: float,
    w_xgb: float,
    w_lgbm: float,
) -> np.ndarray:
    rf_proba = rf_model.predict_proba(X_processed)[:, 1]
    xgb_proba = xgb_model.predict_proba(X_processed)[:, 1]
    lgbm_proba = lgbm_model.predict_proba(X_processed)[:, 1]

    y_proba = (w_rf * rf_proba + w_xgb * xgb_proba + w_lgbm * lgbm_proba)
    return y_proba

def prepare_model_input(
    df: pd.DataFrame,
    raw_input_columns: list[str],
    target_col: str = "Churn",
) -> tuple[pd.DataFrame, Optional[pd.Series]]:
    has_target = target_col in df.columns
    y_true = to_binary_series(df[target_col].copy()) if has_target else None

    df_features = df.drop(columns=[target_col]).copy() if has_target else df.copy()

    metadata_cols = [
        "source",
        "submission_id",
        "submitted_at",
        "promoted_at",
        "created_at_utc",
        "validation_status",
        "validation_errors",
        "firestore_doc_id",
    ]

    for col in metadata_cols:
        if col in df_features.columns and col not in raw_input_columns:
            df_features = df_features.drop(columns=[col])

    missing_for_model = [c for c in raw_input_columns if c not in df_features.columns]
    if raw_input_columns and missing_for_model:
        raise ValueError(f"Missing required feature columns: {missing_for_model}")

    X_raw = df_features[raw_input_columns].copy() if raw_input_columns else df_features.copy()
    return X_raw, y_true
