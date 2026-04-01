import time
import numpy as np
import pandas as pd
import pytest
from frontend_core import validate_submission, write_to_firestore
from app_core import (
    prepare_model_input,
    preprocess_for_ensemble,
    predict_weighted_ensemble,
    risk_tier_from_proba,
    simple_action_recommendations,
)


# ===========================================================================
# ORIGINAL TESTS (18–20)
# ===========================================================================

def test_basic_submission_to_firestore_flow(fake_db, fake_firestore, sample_schema, valid_payload):
    """Verify end-to-end valid submission flow: validate, log, and promote."""
    print("Test 18: Valid submission should validate, log, and promote end-to-end.")
    start = time.perf_counter()
    errors = validate_submission(valid_payload, sample_schema)
    assert errors == []
    submission_id, promoted_id = write_to_firestore(
        db=fake_db, firestore_module=fake_firestore,
        payload=valid_payload, errors=errors, source_value="integration_test",
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    submissions = fake_db.collection("customer_submissions").documents
    current_customers = fake_db.collection("current_customers").documents
    assert submission_id is not None
    assert promoted_id == submission_id
    assert submission_id in submissions
    assert promoted_id in current_customers
    assert current_customers[promoted_id]["CustomerID"] == valid_payload["CustomerID"]
    assert current_customers[promoted_id]["submission_id"] == submission_id
    print(f"  RESULT: Full intake pipeline completed in {elapsed_ms:.2f} ms — validated (0 errors), logged as {submission_id}, promoted to current_customers (CustomerID={valid_payload['CustomerID']})")


def test_basic_dashboard_scoring_flow(
    metadata_dict, mock_preprocessor_array, mock_rf_model, mock_xgb_model, mock_lgbm_model,
):
    """Verify end-to-end scoring flow from raw input to probability, prediction, and risk tier."""
    print("Test 19: Dashboard scoring flow should run end-to-end from raw input to risk tier.")
    raw_input_columns = metadata_dict["raw_input_columns"]
    weights = metadata_dict["weights"]
    model_input_df = pd.DataFrame([{
        "CustomerID": 1001, "Tenure": 12, "PreferredLoginDevice": "Mobile Phone",
        "CityTier": 1, "WarehouseToHome": 15, "PreferredPaymentMode": "Debit Card",
        "Gender": "Male", "HourSpendOnApp": 3, "NumberOfDeviceRegistered": 2,
        "PreferedOrderCat": "Mobile Phone", "SatisfactionScore": 4,
        "MaritalStatus": "Single", "NumberOfAddress": 2, "Complain": 0,
        "OrderAmountHikeFromlastYear": 12, "CouponUsed": 1, "OrderCount": 3,
        "DaySinceLastOrder": 5, "CashbackAmount": 120.0, "Churn": 0,
        "source": "client_submission", "submission_id": "s1", "submitted_at": "t1",
    }])
    start = time.perf_counter()
    X_raw, y = prepare_model_input(model_input_df, raw_input_columns)
    X_proc = preprocess_for_ensemble(X_raw, mock_preprocessor_array, metadata_dict["selected_features"])
    probs = predict_weighted_ensemble(
        X_proc, mock_rf_model, mock_xgb_model, mock_lgbm_model,
        weights["w_rf"], weights["w_xgb"], weights["w_lgbm"],
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    preds = (probs >= 0.5).astype(int)
    tiers = [risk_tier_from_proba(p) for p in probs]
    assert len(probs) == len(X_raw)
    assert len(preds) == len(X_raw)
    assert len(tiers) == len(X_raw)
    assert set(preds.tolist()).issubset({0, 1})
    print(f"  RESULT: Customer 1001 scored in {elapsed_ms:.2f} ms — churn probability={probs[0]:.4f}, prediction={'Churn' if preds[0] else 'Stay'}, risk tier='{tiers[0]}'")


def test_invalid_submission_does_not_promote(fake_db, fake_firestore, sample_schema, valid_payload):
    """Verify invalid submissions are logged but never promoted into current_customers."""
    print("Test 20: Invalid submission should be logged but not promoted.")
    bad_payload = dict(valid_payload); bad_payload["PreferredLoginDevice"] = ""
    start = time.perf_counter()
    errors = validate_submission(bad_payload, sample_schema)
    assert errors != []
    submission_id, promoted_id = write_to_firestore(
        db=fake_db, firestore_module=fake_firestore,
        payload=bad_payload, errors=errors, source_value="integration_test",
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    submissions = fake_db.collection("customer_submissions").documents
    current_customers = fake_db.collection("current_customers").documents
    assert submission_id is not None
    assert promoted_id is None
    assert submission_id in submissions
    assert len(current_customers) == 2
    print(f"  RESULT: Invalid form blocked in {elapsed_ms:.2f} ms — logged as {submission_id} with {len(errors)} error(s), current_customers unchanged at {len(current_customers)} records")


# ===========================================================================
# EXTENDED TESTS (80–86)
# ===========================================================================

def _make_customers(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "CustomerID": range(1001, 1001 + n),
        "Tenure": rng.integers(0, 62, n),
        "PreferredLoginDevice": rng.choice(["Mobile Phone", "Computer"], n),
        "CityTier": rng.integers(1, 4, n),
        "WarehouseToHome": rng.integers(5, 40, n),
        "PreferredPaymentMode": rng.choice(["Debit Card", "Credit Card", "UPI", "Cash on Delivery"], n),
        "Gender": rng.choice(["Male", "Female"], n),
        "HourSpendOnApp": rng.integers(0, 6, n),
        "NumberOfDeviceRegistered": rng.integers(1, 7, n),
        "PreferedOrderCat": rng.choice(["Mobile Phone", "Laptop & Accessory", "Fashion", "Grocery"], n),
        "SatisfactionScore": rng.integers(1, 6, n),
        "MaritalStatus": rng.choice(["Single", "Married", "Divorced"], n),
        "NumberOfAddress": rng.integers(1, 10, n),
        "Complain": rng.integers(0, 2, n),
        "OrderAmountHikeFromlastYear": rng.integers(11, 26, n),
        "CouponUsed": rng.integers(0, 10, n),
        "OrderCount": rng.integers(1, 16, n),
        "DaySinceLastOrder": rng.integers(0, 31, n),
        "CashbackAmount": rng.uniform(0, 300, n).round(2),
        "Churn": rng.integers(0, 2, n),
        "source": ["test"] * n,
        "submission_id": [f"s{i}" for i in range(n)],
        "submitted_at": ["t1"] * n,
    })


_RAW_COLS = [
    "CustomerID", "Tenure", "PreferredLoginDevice", "CityTier", "WarehouseToHome",
    "PreferredPaymentMode", "Gender", "HourSpendOnApp", "NumberOfDeviceRegistered",
    "PreferedOrderCat", "SatisfactionScore", "MaritalStatus", "NumberOfAddress",
    "Complain", "OrderAmountHikeFromlastYear", "CouponUsed", "OrderCount",
    "DaySinceLastOrder", "CashbackAmount",
]


def test_batch_10_customers_correct_output_shapes(
    mock_preprocessor_array, mock_rf_model, mock_xgb_model, mock_lgbm_model
):
    """10-customer batch must produce consistent probabilities, tiers, and predictions."""
    print("Test 80: Scoring batch of 10 customers should produce correct output shapes.")
    df = _make_customers(10)
    start = time.perf_counter()
    X, y = prepare_model_input(df, _RAW_COLS)
    X_proc = preprocess_for_ensemble(X, mock_preprocessor_array, ["f1", "f2", "f3"])
    probs = predict_weighted_ensemble(X_proc, mock_rf_model, mock_xgb_model, mock_lgbm_model, 0.2, 0.3, 0.5)
    elapsed_ms = (time.perf_counter() - start) * 1000
    tiers = [risk_tier_from_proba(float(p)) for p in probs]
    preds = (probs >= 0.5).astype(int)
    assert len(probs) == 10 and len(tiers) == 10 and len(preds) == 10
    assert set(tiers).issubset({"Low", "Medium", "High", "Critical"})
    assert set(preds.tolist()).issubset({0, 1})
    tier_counts = {t: tiers.count(t) for t in ["Low", "Medium", "High", "Critical"] if t in tiers}
    print(f"  RESULT: 10 customers scored in {elapsed_ms:.2f} ms — predicted churners: {int(preds.sum())}/10, tier breakdown: {tier_counts}")


def test_batch_100_customers_all_tiers_valid(
    mock_preprocessor_array, mock_rf_model, mock_xgb_model, mock_lgbm_model
):
    """100-customer batch must produce 100 valid risk tiers."""
    print("Test 81: 100-customer batch should produce 100 valid tiers.")
    df = _make_customers(100)
    cols = [c for c in _RAW_COLS if c in df.columns]
    start = time.perf_counter()
    X, _ = prepare_model_input(df, cols)
    X_proc = preprocess_for_ensemble(X, mock_preprocessor_array, ["f1", "f2", "f3"])
    probs = predict_weighted_ensemble(X_proc, mock_rf_model, mock_xgb_model, mock_lgbm_model, 0.33, 0.33, 0.34)
    elapsed_ms = (time.perf_counter() - start) * 1000
    tiers = [risk_tier_from_proba(float(p)) for p in probs]
    assert len(tiers) == 100
    assert all(t in {"Low", "Medium", "High", "Critical"} for t in tiers)
    tier_counts = {t: tiers.count(t) for t in ["Low", "Medium", "High", "Critical"]}
    print(f"  RESULT: 100 customers scored and tiered in {elapsed_ms:.2f} ms — Low:{tier_counts['Low']}, Medium:{tier_counts['Medium']}, High:{tier_counts['High']}, Critical:{tier_counts['Critical']}")


def test_action_recommendations_generated_for_every_customer(
    mock_preprocessor_array, mock_rf_model, mock_xgb_model, mock_lgbm_model
):
    """Every customer in a 20-customer batch must receive at least 1 recommendation."""
    print("Test 82: Every customer in a 20-customer batch should get >=1 recommendation.")
    df = _make_customers(20)
    start = time.perf_counter()
    all_actions = [simple_action_recommendations(row) for _, row in df.iterrows()]
    elapsed_ms = (time.perf_counter() - start) * 1000
    for actions in all_actions:
        assert len(actions) >= 1
    total_actions = sum(len(a) for a in all_actions)
    avg_actions = total_actions / 20
    print(f"  RESULT: Action recommendations generated for all 20 customers in {elapsed_ms:.2f} ms — avg {avg_actions:.1f} actions per customer")


def test_batch_probs_always_in_valid_range(
    mock_preprocessor_array, mock_rf_model, mock_xgb_model, mock_lgbm_model
):
    """All predicted probabilities for a 50-customer batch must sit in [0.0, 1.0]."""
    print("Test 83: All probabilities should be within [0.0, 1.0].")
    df = _make_customers(50)
    cols = [c for c in _RAW_COLS if c in df.columns]
    start = time.perf_counter()
    X, _ = prepare_model_input(df, cols)
    X_proc = preprocess_for_ensemble(X, mock_preprocessor_array, ["f1", "f2", "f3"])
    probs = predict_weighted_ensemble(X_proc, mock_rf_model, mock_xgb_model, mock_lgbm_model, 0.2, 0.3, 0.5)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
    print(f"  RESULT: 50 churn probabilities all valid in {elapsed_ms:.2f} ms — range [{probs.min():.3f}, {probs.max():.3f}], avg {probs.mean():.3f}")


def test_valid_submission_full_pipeline(
    fake_db, fake_firestore, sample_schema, valid_payload,
    mock_preprocessor_array, mock_rf_model, mock_xgb_model, mock_lgbm_model,
):
    """Validate → write → score a single valid customer end-to-end."""
    print("Test 84: Full pipeline for a valid customer should complete without errors.")
    start = time.perf_counter()
    errors = validate_submission(valid_payload, sample_schema)
    assert errors == []
    sid, pid = write_to_firestore(
        db=fake_db, firestore_module=fake_firestore,
        payload=valid_payload, errors=errors, source_value="integration_test",
    )
    assert pid == sid
    assert sid in fake_db.collection("current_customers").documents
    df = pd.DataFrame([valid_payload])
    cols = [c for c in ["CustomerID", "Tenure", "CityTier", "Complain"] if c in df.columns]
    X, _ = prepare_model_input(df, cols)
    X_proc = preprocess_for_ensemble(X, mock_preprocessor_array, ["f1", "f2", "f3"])
    probs = predict_weighted_ensemble(X_proc, mock_rf_model, mock_xgb_model, mock_lgbm_model, 0.2, 0.3, 0.5)
    elapsed_ms = (time.perf_counter() - start) * 1000
    tier = risk_tier_from_proba(float(probs[0]))
    assert tier in {"Low", "Medium", "High", "Critical"}
    print(f"  RESULT: Full pipeline (intake → Firestore → score) completed in {elapsed_ms:.2f} ms — CustomerID={valid_payload['CustomerID']}, churn prob={probs[0]:.4f}, risk tier='{tier}'")


def test_invalid_submission_blocked_from_current_customers(
    fake_db, fake_firestore, sample_schema, valid_payload
):
    """Invalid submission must not be promoted to current_customers."""
    print("Test 85: Invalid submission should not reach current_customers.")
    bad_payload = dict(valid_payload); bad_payload["PreferredLoginDevice"] = ""
    start = time.perf_counter()
    errors = validate_submission(bad_payload, sample_schema)
    assert errors != []
    sid, pid = write_to_firestore(
        db=fake_db, firestore_module=fake_firestore,
        payload=bad_payload, errors=errors, source_value="integration_test",
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert sid is not None and pid is None
    assert len(fake_db.collection("current_customers").documents) == 2
    print(f"  RESULT: Invalid submission blocked in {elapsed_ms:.2f} ms — {len(errors)} validation error(s), current_customers unchanged at 2 records")


def test_mixed_batch_five_submissions_correct_partition(
    fake_db, fake_firestore, sample_schema, valid_payload
):
    """3 valid + 2 invalid → submissions=5, current_customers=2 seeds + 3 promoted."""
    print("Test 86: 3 valid + 2 invalid submissions should partition correctly.")
    start = time.perf_counter()
    for i in range(3):
        p = dict(valid_payload); p["CustomerID"] = 2000 + i
        errors = validate_submission(p, sample_schema)
        write_to_firestore(db=fake_db, firestore_module=fake_firestore,
                           payload=p, errors=errors, source_value="test")
    for i in range(2):
        p = dict(valid_payload); p["CustomerID"] = 3000 + i
        p["PreferredLoginDevice"] = "Tablet"
        errors = validate_submission(p, sample_schema)
        write_to_firestore(db=fake_db, firestore_module=fake_firestore,
                           payload=p, errors=errors, source_value="test")
    elapsed_ms = (time.perf_counter() - start) * 1000
    subs = len(fake_db.collection("customer_submissions").documents)
    curr = len(fake_db.collection("current_customers").documents)
    assert subs == 5
    assert curr == 5
    print(f"  RESULT: 5 submissions processed in {elapsed_ms:.2f} ms — customer_submissions={subs} (all logged), current_customers={curr} (2 seeds + 3 valid promoted, 2 invalid blocked)")