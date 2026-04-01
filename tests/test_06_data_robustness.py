"""
test_data_robustness.py

Tests that the pipeline handles messy, edge-case, or unusual data gracefully.
"""

from __future__ import annotations

import time
import numpy as np
import pandas as pd
import pytest

from app_core import (
    predict_weighted_ensemble,
    prepare_model_input,
    preprocess_for_ensemble,
    to_binary_series,
)
from frontend_core import validate_submission


def test_robustness_binary_series_empty_no_error():
    """Empty series must return an empty series without raising (Test 111)."""
    print("Test 111: Empty binary series should return empty without error.")
    start = time.perf_counter()
    out = to_binary_series(pd.Series([], dtype=object))
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert len(out) == 0
    print(f"  RESULT: Empty Churn column handled in {elapsed_ms:.3f} ms — dashboard won't crash on empty dataset upload")


def test_robustness_binary_series_all_none():
    """Series of None values must all map to 0 (Test 112)."""
    print("Test 112: All-None series should map to 0.")
    s = pd.Series([None, None, None])
    start = time.perf_counter()
    out = to_binary_series(s)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert (out == 0).all()
    print(f"  RESULT: 3 null Churn values safely defaulted to 0 in {elapsed_ms:.3f} ms — missing labels treated as non-churn")


def test_robustness_binary_series_large_mixed_correct_counts():
    """50 000-row mixed series must produce exactly 25 000 ones and 25 000 zeros (Test 113)."""
    print("Test 113: Large mixed series should produce exact 1/0 counts.")
    s = pd.Series(["Yes", "No", "1", "0"] * 12_500)
    start = time.perf_counter()
    out = to_binary_series(s)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert len(out) == 50_000
    assert (out == 1).sum() == 25_000
    assert (out == 0).sum() == 25_000
    print(f"  RESULT: 50,000 mixed-format Churn labels encoded in {elapsed_ms:.2f} ms — 25,000 churned / 25,000 retained, no data loss")


def test_robustness_prepare_single_row_correct_shape():
    """Single-row DataFrame must produce shape (1, n) for X (Test 114)."""
    print("Test 114: Single-row input should produce correct shape (1, n).")
    df = pd.DataFrame({"CustomerID": [1], "Tenure": [5], "Churn": [1]})
    start = time.perf_counter()
    X, y = prepare_model_input(df, ["CustomerID", "Tenure"])
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert X.shape == (1, 2)
    assert y.tolist() == [1]
    print(f"  RESULT: Single customer row prepared in {elapsed_ms:.3f} ms — X shape {X.shape}, Churn={y.tolist()[0]} (dashboard can score individual customers)")


def test_robustness_prepare_all_metadata_columns_removed():
    """All 8 metadata columns must be excluded from X (Test 115)."""
    print("Test 115: All metadata columns should be excluded from model input X.")
    df = pd.DataFrame({
        "CustomerID": [1, 2], "Tenure": [5, 10], "Churn": [0, 1],
        "source": ["a", "b"], "submission_id": ["s1", "s2"],
        "submitted_at": ["t1", "t2"], "promoted_at": ["p1", "p2"],
        "created_at_utc": ["c1", "c2"], "validation_status": ["valid", "valid"],
        "validation_errors": [[], []], "firestore_doc_id": ["d1", "d2"],
    })
    start = time.perf_counter()
    X, _ = prepare_model_input(df, ["CustomerID", "Tenure"])
    elapsed_ms = (time.perf_counter() - start) * 1000
    meta_cols = ["source", "submission_id", "submitted_at", "promoted_at",
                 "created_at_utc", "validation_status", "validation_errors", "firestore_doc_id"]
    for col in meta_cols:
        assert col not in X.columns
    print(f"  RESULT: All {len(meta_cols)} Firestore metadata columns stripped in {elapsed_ms:.3f} ms — model only sees {list(X.columns)}")


def test_robustness_prepare_churn_float_encoded_correctly():
    """Churn as float 1.0/0.0 must encode correctly to int 1/0 (Test 116)."""
    print("Test 116: Float Churn values 1.0/0.0 should encode to int 1/0.")
    df = pd.DataFrame({"A": [1, 2], "Churn": [1.0, 0.0]})
    start = time.perf_counter()
    _, y = prepare_model_input(df, ["A"])
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert y.tolist() == [1, 0]
    print(f"  RESULT: Float Churn labels {[1.0, 0.0]} encoded to int {y.tolist()} in {elapsed_ms:.3f} ms")


def test_robustness_prepare_missing_required_columns_raises(sample_raw_df):
    """Missing required columns must raise a ValueError with 'missing' in the message (Test 117)."""
    print("Test 117: Missing required columns should raise ValueError.")
    broken = sample_raw_df.drop(columns=["Tenure"])
    start = time.perf_counter()
    with pytest.raises(ValueError, match="(?i)missing"):
        prepare_model_input(broken, ["CustomerID", "Tenure", "PreferredLoginDevice", "CityTier", "Complain"])
    elapsed_ms = (time.perf_counter() - start) * 1000
    print(f"  RESULT: Missing 'Tenure' column caught and raised in {elapsed_ms:.3f} ms — dashboard will show error instead of scoring silently on bad data")


def test_robustness_preprocess_always_returns_dataframe(mock_preprocessor_array):
    """preprocess_for_ensemble must always return a pd.DataFrame type (Test 118)."""
    print("Test 118: preprocess_for_ensemble should always return a pd.DataFrame.")
    df = pd.DataFrame({"A": [1, 2, 3]})
    start = time.perf_counter()
    out = preprocess_for_ensemble(df, mock_preprocessor_array, ["f1", "f2", "f3"])
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert isinstance(out, pd.DataFrame)
    print(f"  RESULT: Preprocessing returned {type(out).__name__} with shape {out.shape} in {elapsed_ms:.3f} ms — scoring pipeline always receives correct type")


def test_robustness_preprocess_500_rows_correct_shape(mock_preprocessor_array):
    """500-row input must produce a 500-row output (Test 119)."""
    print("Test 119: 500-row input should produce 500-row output.")
    df = pd.DataFrame({"A": range(500), "B": range(500)})
    start = time.perf_counter()
    out = preprocess_for_ensemble(df, mock_preprocessor_array, ["f1", "f2", "f3"])
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert out.shape == (500, 3)
    print(f"  RESULT: 500 rows → output shape {out.shape} in {elapsed_ms:.2f} ms — no rows dropped during preprocessing")


def test_robustness_predict_single_row(mock_rf_model, mock_xgb_model, mock_lgbm_model):
    """Single-row prediction must return one value in [0, 1] (Test 120)."""
    print("Test 120: Single-row prediction should return one valid probability.")
    X = pd.DataFrame({"f1": [0.5]})
    start = time.perf_counter()
    out = predict_weighted_ensemble(X, mock_rf_model, mock_xgb_model, mock_lgbm_model, 0.2, 0.3, 0.5)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert len(out) == 1
    assert 0.0 <= float(out[0]) <= 1.0
    print(f"  RESULT: Single customer churn probability = {out[0]:.4f} in {elapsed_ms:.3f} ms — individual customer lookup tab works correctly")


def test_robustness_predict_large_batch_no_nan(mock_rf_model, mock_xgb_model, mock_lgbm_model):
    """1 000-row prediction must produce no NaN values in output (Test 121)."""
    print("Test 121: 1,000-row prediction should produce no NaN values.")
    X = pd.DataFrame({"f": np.random.rand(1_000)})
    start = time.perf_counter()
    out = predict_weighted_ensemble(X, mock_rf_model, mock_xgb_model, mock_lgbm_model, 0.33, 0.33, 0.34)
    elapsed_ms = (time.perf_counter() - start) * 1000
    nan_count = int(np.isnan(out).sum())
    assert nan_count == 0
    print(f"  RESULT: 1,000 customers scored in {elapsed_ms:.2f} ms — {nan_count} NaN values (dashboard churn probability column fully populated)")


def test_robustness_validate_extra_keys_no_errors(sample_schema, valid_payload):
    """Extra payload keys not in schema must not cause any errors (Test 122)."""
    print("Test 122: Extra payload keys outside the schema should be ignored.")
    payload = dict(valid_payload); payload["ExtraFieldNotInSchema"] = "surprise"
    start = time.perf_counter()
    errors = validate_submission(payload, sample_schema)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert errors == []
    print(f"  RESULT: Extra field silently ignored in {elapsed_ms:.3f} ms — intake form is robust to unexpected browser-injected fields")


def test_robustness_validate_empty_payload_reports_all_required(sample_schema):
    """Completely empty payload must flag every required column (Test 123)."""
    print("Test 123: Empty payload should report errors for all required columns.")
    start = time.perf_counter()
    errors = validate_submission({}, sample_schema)
    elapsed_ms = (time.perf_counter() - start) * 1000
    error_text = " ".join(errors)
    required_cols = [col for col in sample_schema["numeric_cols"] if col != "Churn"]
    for col in required_cols:
        assert col in error_text, f"Expected error for required column '{col}'"
    print(f"  RESULT: Empty form submission caught {len(errors)} missing field errors in {elapsed_ms:.3f} ms — all {len(required_cols)} required fields flagged")