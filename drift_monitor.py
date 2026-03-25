from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

try:
    import firebase_admin
    from firebase_admin import credentials, firestore

    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False


NUMERIC_FEATURES = [
    "Tenure",
    "WarehouseToHome",
    "HourSpendOnApp",
    "NumberOfDeviceRegistered",
    "SatisfactionScore",
    "NumberOfAddress",
    "OrderAmountHikeFromlastYear",
    "CouponUsed",
    "OrderCount",
    "DaySinceLastOrder",
    "CashbackAmount",
]

CATEGORICAL_FEATURES = [
    "PreferredLoginDevice",
    "PreferredPaymentMode",
    "Gender",
    "PreferedOrderCat",
    "MaritalStatus",
]

EPS = 1e-6


def load_reference_distribution(csv_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def load_current_distribution(
    current_df: Optional[pd.DataFrame] = None,
    service_account_path: Optional[str] = None,
    collection_name: str = "current_customers",
) -> pd.DataFrame:
    if current_df is not None:
        return current_df.copy()

    if not FIREBASE_AVAILABLE:
        raise RuntimeError("firebase-admin is not installed.")
    if not service_account_path:
        raise ValueError("service_account_path is required when current_df is not provided.")

    if not firebase_admin._apps:
        cred = credentials.Certificate(service_account_path)
        firebase_admin.initialize_app(cred)

    db = firestore.client()
    rows = []
    # Add limit to prevent excessive reads when fallback is used
    for doc in db.collection(collection_name).limit(10000).stream():
        rows.append(doc.to_dict() or {})

    return pd.DataFrame(rows)


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").dropna()


def compute_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    ref = _safe_numeric(reference)
    cur = _safe_numeric(current)

    if ref.empty or cur.empty:
        return float("nan")

    ref_min = float(ref.min())
    ref_max = float(ref.max())

    if ref_min == ref_max:
        return 0.0

    bin_edges = np.linspace(ref_min, ref_max, bins + 1)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref.values, bins=bin_edges)
    cur_counts, _ = np.histogram(cur.values, bins=bin_edges)

    ref_dist = ref_counts / max(ref_counts.sum(), 1)
    cur_dist = cur_counts / max(cur_counts.sum(), 1)

    ref_dist = np.where(ref_dist == 0, EPS, ref_dist)
    cur_dist = np.where(cur_dist == 0, EPS, cur_dist)

    return float(np.sum((cur_dist - ref_dist) * np.log(cur_dist / ref_dist)))


def compute_categorical_chi2_pvalue(reference: pd.Series, current: pd.Series) -> float:
    ref = reference.dropna().astype(str)
    cur = current.dropna().astype(str)

    if ref.empty or cur.empty:
        return float("nan")

    categories = sorted(set(ref.unique()).union(set(cur.unique())))
    if len(categories) <= 1:
        return 1.0

    ref_counts = ref.value_counts().reindex(categories, fill_value=0)
    cur_counts = cur.value_counts().reindex(categories, fill_value=0)

    contingency = np.vstack([ref_counts.values, cur_counts.values])
    if contingency.sum() == 0:
        return float("nan")

    _, p_value, _, _ = chi2_contingency(contingency)
    return float(p_value)


def build_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for feature in NUMERIC_FEATURES:
        if feature not in reference_df.columns or feature not in current_df.columns:
            rows.append(
                {
                    "feature": feature,
                    "type": "numeric",
                    "statistic": float("nan"),
                    "threshold": "PSI > 0.2 Drift; > 0.1 Warning",
                    "status": "Warning",
                }
            )
            continue

        psi = compute_psi(reference_df[feature], current_df[feature], bins=10)

        if np.isnan(psi):
            status = "Warning"
        elif psi > 0.2:
            status = "Drift"
        elif psi > 0.1:
            status = "Warning"
        else:
            status = "OK"

        rows.append(
            {
                "feature": feature,
                "type": "numeric",
                "statistic": psi,
                "threshold": "PSI > 0.2 Drift; > 0.1 Warning",
                "status": status,
            }
        )

    for feature in CATEGORICAL_FEATURES:
        if feature not in reference_df.columns or feature not in current_df.columns:
            rows.append(
                {
                    "feature": feature,
                    "type": "categorical",
                    "statistic": float("nan"),
                    "threshold": "p < 0.05 Warning",
                    "status": "Warning",
                }
            )
            continue

        p_value = compute_categorical_chi2_pvalue(reference_df[feature], current_df[feature])

        if np.isnan(p_value):
            status = "Warning"
        elif p_value < 0.05:
            status = "Warning"
        else:
            status = "OK"

        rows.append(
            {
                "feature": feature,
                "type": "categorical",
                "statistic": p_value,
                "threshold": "p < 0.05 Warning",
                "status": status,
            }
        )

    return pd.DataFrame(rows, columns=["feature", "type", "statistic", "threshold", "status"])


def save_drift_report(report_df: pd.DataFrame, path: str | Path) -> None:
    report_with_ts = report_df.copy()
    report_with_ts["timestamp"] = datetime.now(timezone.utc).isoformat()

    output_path = Path(path)
    if output_path.exists():
        existing = pd.read_csv(output_path)
        combined = pd.concat([existing, report_with_ts], ignore_index=True)
        combined.to_csv(output_path, index=False)
    else:
        report_with_ts.to_csv(output_path, index=False)


def run_drift_monitor(
    reference_csv_path: str | Path,
    current_df: Optional[pd.DataFrame] = None,
    service_account_path: Optional[str] = None,
    collection_name: str = "current_customers",
    output_path: str | Path = "drift_report.csv",
) -> pd.DataFrame:
    reference_df = load_reference_distribution(reference_csv_path)
    current_live_df = load_current_distribution(
        current_df=current_df,
        service_account_path=service_account_path,
        collection_name=collection_name,
    )

    report_df = build_drift_report(reference_df, current_live_df)
    save_drift_report(report_df, output_path)
    return report_df
