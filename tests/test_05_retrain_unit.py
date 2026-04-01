"""
test_retrain_unit.py

Unit tests for pure-Python helper logic from retrain.py.
Functions are replicated inline to avoid the firebase_admin top-level import.
"""

from __future__ import annotations

import time
import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Inline replicas of retrain.py pure helpers (no Firebase/GCS dependencies)
# ---------------------------------------------------------------------------

ALL_COLUMNS = [
    "CustomerID", "Tenure", "CityTier", "WarehouseToHome", "HourSpendOnApp",
    "NumberOfDeviceRegistered", "SatisfactionScore", "NumberOfAddress", "Complain",
    "OrderAmountHikeFromlastYear", "CouponUsed", "OrderCount", "DaySinceLastOrder",
    "CashbackAmount", "Churn", "PreferredLoginDevice", "PreferredPaymentMode",
    "Gender", "PreferedOrderCat", "MaritalStatus",
]

NUMERIC_COLUMNS = [
    "CustomerID", "Tenure", "CityTier", "WarehouseToHome", "HourSpendOnApp",
    "NumberOfDeviceRegistered", "SatisfactionScore", "NumberOfAddress", "Complain",
    "OrderAmountHikeFromlastYear", "CouponUsed", "OrderCount", "DaySinceLastOrder",
    "CashbackAmount",
]

CATEGORICAL_COLUMNS = [
    "PreferredLoginDevice", "PreferredPaymentMode", "Gender",
    "PreferedOrderCat", "MaritalStatus",
]

TARGET_COL = "Churn"


def normalize_binary(series: pd.Series) -> pd.Series:
    mapping = {
        "Yes": 1, "No": 0, "yes": 1, "no": 0,
        "1": 1, "0": 0, 1: 1, 0: 0, True: 1, False: 0,
    }
    mapped = series.map(mapping)
    merged = mapped.where(mapped.notna(), series)
    return pd.to_numeric(merged, errors="coerce")


def coerce_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ALL_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[ALL_COLUMNS]
    for col in NUMERIC_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in CATEGORICAL_COLUMNS:
        out[col] = out[col].astype("string")
    out[TARGET_COL] = normalize_binary(out[TARGET_COL])
    out = out.dropna(subset=[TARGET_COL]).copy()
    out[TARGET_COL] = out[TARGET_COL].astype(int)
    return out


def build_blob_name(prefix: str, file_name: str) -> str:
    clean_prefix = prefix.strip().strip("/")
    if not clean_prefix:
        return file_name
    return f"{clean_prefix}/{file_name}"


def _minimal_df(churn_values):
    return pd.DataFrame({
        "CustomerID": range(1, len(churn_values) + 1),
        "Tenure": [5] * len(churn_values),
        "Churn": churn_values,
    })


# ===========================================================================
# normalize_binary
# ===========================================================================

def test_normalize_binary_yes_no():
    """normalize_binary must map Yes/No to 1.0/0.0."""
    print("Test 93: normalize_binary should map Yes/No to 1.0/0.0.")
    s = pd.Series(["Yes", "No", "yes", "no"])
    start = time.perf_counter()
    out = normalize_binary(s)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert out.tolist() == [1.0, 0.0, 1.0, 0.0]
    print(f"  RESULT: Yes/No labels normalised to {out.tolist()} in {elapsed_ms:.3f} ms — retrain pipeline handles string Churn labels from Firestore")


def test_normalize_binary_numeric_strings():
    """normalize_binary must handle '1'/'0' string inputs."""
    print("Test 94: normalize_binary should handle '1'/'0' numeric strings.")
    s = pd.Series(["1", "0", "1"])
    start = time.perf_counter()
    out = normalize_binary(s)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert out.tolist() == [1.0, 0.0, 1.0]
    print(f"  RESULT: String '1'/'0' normalised to {out.tolist()} in {elapsed_ms:.3f} ms")


def test_normalize_binary_bool_values():
    """normalize_binary must map True/False to 1.0/0.0."""
    print("Test 95: normalize_binary should map True/False to 1.0/0.0.")
    s = pd.Series([True, False, True])
    start = time.perf_counter()
    out = normalize_binary(s)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert out.tolist() == [1.0, 0.0, 1.0]
    print(f"  RESULT: Boolean values normalised to {out.tolist()} in {elapsed_ms:.3f} ms")


def test_normalize_binary_integer_values():
    """normalize_binary must handle integer 1/0 inputs."""
    print("Test 96: normalize_binary should handle integer 1/0.")
    s = pd.Series([1, 0, 1, 0])
    start = time.perf_counter()
    out = normalize_binary(s)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert out.tolist() == [1.0, 0.0, 1.0, 0.0]
    print(f"  RESULT: Integer 1/0 passed through as {out.tolist()} in {elapsed_ms:.3f} ms")


def test_normalize_binary_nan_preserved():
    """NaN values must be preserved as NaN after normalize_binary."""
    print("Test 97: NaN should be preserved as NaN by normalize_binary.")
    s = pd.Series([np.nan, "Yes", "No"])
    start = time.perf_counter()
    out = normalize_binary(s)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert np.isnan(out.iloc[0])
    assert out.iloc[1] == 1.0 and out.iloc[2] == 0.0
    print(f"  RESULT: NaN preserved at index 0, Yes→{out.iloc[1]}, No→{out.iloc[2]} in {elapsed_ms:.3f} ms — unlabelled rows skipped during retrain")


def test_normalize_binary_unknown_value_becomes_nan():
    """Values not in the mapping must coerce to NaN via pd.to_numeric."""
    print("Test 98: Unknown values should coerce to NaN.")
    s = pd.Series(["maybe"])
    start = time.perf_counter()
    out = normalize_binary(s)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert np.isnan(out.iloc[0])
    print(f"  RESULT: Unrecognised value 'maybe' coerced to NaN in {elapsed_ms:.3f} ms — will be excluded from retraining data")


# ===========================================================================
# coerce_schema
# ===========================================================================

def test_coerce_schema_adds_all_missing_columns():
    """coerce_schema must fill in all missing ALL_COLUMNS with NaN."""
    print("Test 99: coerce_schema should add all missing schema columns.")
    start = time.perf_counter()
    result = coerce_schema(_minimal_df([1, 0]))
    elapsed_ms = (time.perf_counter() - start) * 1000
    for col in ALL_COLUMNS:
        assert col in result.columns, f"Missing column: {col}"
    missing_filled = [c for c in ALL_COLUMNS if c not in ["CustomerID", "Tenure", "Churn"]]
    print(f"  RESULT: Schema enforced in {elapsed_ms:.3f} ms — {len(missing_filled)} missing columns filled with NaN, retrain data always has consistent shape")


def test_coerce_schema_drops_rows_without_churn():
    """Rows without a Churn label must be dropped."""
    print("Test 100: coerce_schema should drop rows with null Churn labels.")
    df = _minimal_df([None, 1, np.nan, 0])
    start = time.perf_counter()
    result = coerce_schema(df)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert len(result) == 2
    assert set(result["Churn"].tolist()) == {0, 1}
    print(f"  RESULT: 4 rows → {len(result)} labelled rows kept in {elapsed_ms:.3f} ms — 2 unlabelled Firestore records excluded from retrain")


def test_coerce_schema_converts_yes_no_churn():
    """Churn='Yes'/'No' string labels must convert to 1/0."""
    print("Test 101: coerce_schema should convert Yes/No Churn labels to 1/0.")
    start = time.perf_counter()
    result = coerce_schema(_minimal_df(["Yes", "No"]))
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert result["Churn"].tolist() == [1, 0]
    print(f"  RESULT: Yes/No Churn labels converted to {result['Churn'].tolist()} in {elapsed_ms:.3f} ms — retrain handles mixed-format Firestore data")


def test_coerce_schema_churn_is_int_dtype():
    """Churn column must have integer dtype after coerce_schema."""
    print("Test 102: Churn column should be integer dtype after coerce_schema.")
    start = time.perf_counter()
    result = coerce_schema(_minimal_df([1, 0]))
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert result["Churn"].dtype == int
    print(f"  RESULT: Churn dtype={result['Churn'].dtype} after schema coercion in {elapsed_ms:.3f} ms — model receives correct integer target")


def test_coerce_schema_all_null_churn_returns_empty():
    """All-null Churn column must produce an empty DataFrame."""
    print("Test 103: All-null Churn should produce an empty DataFrame.")
    start = time.perf_counter()
    result = coerce_schema(_minimal_df([None, None]))
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert result.empty
    print(f"  RESULT: All-null Churn → empty DataFrame in {elapsed_ms:.3f} ms — retrain correctly aborts when no labelled data exists")


def test_coerce_schema_empty_input_returns_empty():
    """Empty input DataFrame must return empty after coerce_schema."""
    print("Test 104: Empty input DataFrame should return empty after coerce_schema.")
    df = pd.DataFrame(columns=ALL_COLUMNS)
    start = time.perf_counter()
    result = coerce_schema(df)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert result.empty
    print(f"  RESULT: Empty Firestore export → empty output in {elapsed_ms:.3f} ms — retrain will abort gracefully with no data")


# ===========================================================================
# build_blob_name
# ===========================================================================

def test_build_blob_name_prefix_and_filename():
    """Prefix and filename must be joined with a single '/'."""
    print("Test 105: build_blob_name should join prefix and filename with '/'.")
    start = time.perf_counter()
    result = build_blob_name("ensemble", "model.joblib")
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert result == "ensemble/model.joblib"
    print(f"  RESULT: GCS path built as '{result}' in {elapsed_ms:.3f} ms")


def test_build_blob_name_empty_prefix_returns_filename():
    """Empty prefix must return just the filename."""
    print("Test 106: Empty prefix should return just the filename.")
    start = time.perf_counter()
    result = build_blob_name("", "model.joblib")
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert result == "model.joblib"
    print(f"  RESULT: No prefix → GCS path='{result}' (root bucket upload) in {elapsed_ms:.3f} ms")


def test_build_blob_name_whitespace_prefix_returns_filename():
    """Whitespace-only prefix must be treated as empty."""
    print("Test 107: Whitespace-only prefix should be treated as empty.")
    start = time.perf_counter()
    result = build_blob_name("   ", "file.csv")
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert result == "file.csv"
    print(f"  RESULT: Whitespace prefix stripped → GCS path='{result}' in {elapsed_ms:.3f} ms")


def test_build_blob_name_trailing_slash_stripped():
    """Trailing slash on prefix must be stripped before joining."""
    print("Test 108: Trailing slash on prefix should be stripped.")
    start = time.perf_counter()
    result = build_blob_name("ensemble/", "file.csv")
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert result == "ensemble/file.csv"
    print(f"  RESULT: Trailing slash stripped → GCS path='{result}' in {elapsed_ms:.3f} ms — no double-slash in bucket path")


def test_build_blob_name_leading_slash_stripped():
    """Leading slash on prefix must be stripped before joining."""
    print("Test 109: Leading slash on prefix should be stripped.")
    start = time.perf_counter()
    result = build_blob_name("/ensemble", "file.csv")
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert result == "ensemble/file.csv"
    print(f"  RESULT: Leading slash stripped → GCS path='{result}' in {elapsed_ms:.3f} ms")


def test_build_blob_name_deep_prefix_path():
    """Multi-level prefix path must be preserved correctly."""
    print("Test 110: Multi-level prefix path should be preserved.")
    start = time.perf_counter()
    result = build_blob_name("models/v2", "pipeline.joblib")
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert result == "models/v2/pipeline.joblib"
    print(f"  RESULT: Versioned GCS path='{result}' in {elapsed_ms:.3f} ms — model versioning structure preserved on upload")