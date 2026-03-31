"""
test_performance.py

Timing benchmarks for hot-path dashboard functions.
Run with -s to see the printed timing results:
    pytest tests/test_performance.py -v -s
    pytest tests/test_performance.py -v -s --durations=10
"""

from __future__ import annotations

import time
import numpy as np
import pandas as pd
import pytest

from app_core import (
    predict_weighted_ensemble,
    preprocess_for_ensemble,
    risk_tier_from_proba,
    to_binary_series,
)
from frontend_core import validate_submission, write_to_firestore

T_VALIDATE_PER_CALL_MS  = 10
T_PREPROCESS_500_MS     = 500
T_PREDICT_500_MS        = 300
T_WRITE_FIRESTORE_MS    = 50
T_RISK_TIER_10K_MS      = 2_000
T_BINARY_SERIES_50K_MS  = 1_000


def test_perf_validate_submission_per_call(sample_schema, valid_payload):
    """Customer intake form validation must stay under threshold per call (Test 87)."""
    print("Test 87 PERF: Intake form validation speed (simulates user submitting form).")
    start = time.perf_counter()
    for _ in range(100):
        validate_submission(valid_payload, sample_schema)
    avg_ms = (time.perf_counter() - start) * 1000 / 100
    assert avg_ms < T_VALIDATE_PER_CALL_MS, (
        f"Form validation too slow: {avg_ms:.2f} ms (limit {T_VALIDATE_PER_CALL_MS} ms)"
    )
    print(f"  RESULT: Intake form validates in avg {avg_ms:.3f} ms per submission (limit {T_VALIDATE_PER_CALL_MS} ms) — form responds instantly on submit click")


def test_perf_preprocess_500_rows(mock_preprocessor_array):
    """Dashboard preprocessing pipeline for 500 customers must stay under threshold (Test 88)."""
    print("Test 88 PERF: Dashboard data preprocessing speed (simulates loading 500 customers).")
    df = pd.DataFrame({"A": range(500), "B": range(500)})
    start = time.perf_counter()
    out = preprocess_for_ensemble(df, mock_preprocessor_array, ["f1", "f2", "f3"])
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms < T_PREPROCESS_500_MS
    assert out.shape == (500, 3)
    print(f"  RESULT: 500 customer rows preprocessed in {elapsed_ms:.2f} ms (limit {T_PREPROCESS_500_MS} ms) — dashboard data tab loads in under {T_PREPROCESS_500_MS} ms")


def test_perf_ensemble_predict_500_rows(mock_rf_model, mock_xgb_model, mock_lgbm_model):
    """Ensemble scoring of 500 customers must stay under threshold (Test 89)."""
    print("Test 89 PERF: Ensemble churn scoring speed (simulates scoring 500 customers on page load).")
    X = pd.DataFrame({"f1": np.random.rand(500), "f2": np.random.rand(500)})
    start = time.perf_counter()
    probs = predict_weighted_ensemble(
        X, mock_rf_model, mock_xgb_model, mock_lgbm_model, 0.2, 0.3, 0.5
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms < T_PREDICT_500_MS
    assert len(probs) == 500
    predicted_churners = int((probs >= 0.5).sum())
    print(f"  RESULT: 500 customers scored in {elapsed_ms:.2f} ms (limit {T_PREDICT_500_MS} ms) — {predicted_churners} predicted churners, avg churn prob {probs.mean():.3f}")


def test_perf_write_to_firestore_single(fake_db, fake_firestore, valid_payload):
    """Single Firestore write on form submit must stay under threshold (Test 90)."""
    print("Test 90 PERF: Firestore write speed (simulates user clicking Submit on intake form).")
    start = time.perf_counter()
    sid, pid = write_to_firestore(
        db=fake_db, firestore_module=fake_firestore,
        payload=valid_payload, errors=[], source_value="perf_test",
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms < T_WRITE_FIRESTORE_MS, (
        f"Firestore write too slow: {elapsed_ms:.2f} ms (limit {T_WRITE_FIRESTORE_MS} ms)"
    )
    print(f"  RESULT: Form submit + Firestore write completed in {elapsed_ms:.3f} ms (limit {T_WRITE_FIRESTORE_MS} ms) — submission_id={sid}, promoted_id={pid}")


def test_perf_risk_tier_10k_calls():
    """Risk tier classification for 10 000 probability values must stay under threshold (Test 91)."""
    print("Test 91 PERF: Risk tier rendering speed (simulates scrolling through 10,000 customer rows on dashboard).")
    probas = np.linspace(0.0, 1.0, 10_000)
    start = time.perf_counter()
    tiers = [risk_tier_from_proba(float(p)) for p in probas]
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms < T_RISK_TIER_10K_MS, (
        f"Risk tier classification too slow: {elapsed_ms:.2f} ms (limit {T_RISK_TIER_10K_MS} ms)"
    )
    counts = {t: tiers.count(t) for t in ["Low", "Medium", "High", "Critical"]}
    print(f"  RESULT: 10,000 risk tier badges rendered in {elapsed_ms:.2f} ms (limit {T_RISK_TIER_10K_MS} ms) — Low:{counts['Low']}, Medium:{counts['Medium']}, High:{counts['High']}, Critical:{counts['Critical']}")


def test_perf_to_binary_series_50k_rows():
    """Binary encoding of 50 000 Churn column values must stay under threshold (Test 92)."""
    print("Test 92 PERF: Churn label encoding speed (simulates loading full dataset into dashboard).")
    s = pd.Series(["Yes", "No", "1", "0"] * 12_500)
    start = time.perf_counter()
    out = to_binary_series(s)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms < T_BINARY_SERIES_50K_MS, (
        f"Binary encoding too slow: {elapsed_ms:.2f} ms (limit {T_BINARY_SERIES_50K_MS} ms)"
    )
    assert len(out) == 50_000
    ones = int((out == 1).sum())
    zeros = int((out == 0).sum())
    print(f"  RESULT: 50,000 Churn labels encoded in {elapsed_ms:.2f} ms (limit {T_BINARY_SERIES_50K_MS} ms) — {ones} churned, {zeros} retained")
