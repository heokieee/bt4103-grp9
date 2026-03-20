from frontend_core import validate_submission, write_to_firestore
from app_core import (
    prepare_model_input,
    preprocess_for_ensemble,
    predict_weighted_ensemble,
    risk_tier_from_proba,
)


def test_basic_submission_to_firestore_flow(fake_db, fake_firestore, sample_schema, valid_payload):
    """Verify end-to-end valid submission flow: validate, log, and promote."""
    print("Test 18: Valid submission should validate, log, and promote end-to-end.")
    errors = validate_submission(valid_payload, sample_schema)
    assert errors == []

    submission_id, promoted_id = write_to_firestore(
        db=fake_db,
        firestore_module=fake_firestore,
        payload=valid_payload,
        errors=errors,
        source_value="integration_test",
    )

    submissions = fake_db.collection("customer_submissions").documents
    current_customers = fake_db.collection("current_customers").documents

    assert submission_id is not None
    assert promoted_id == submission_id
    assert submission_id in submissions
    assert promoted_id in current_customers
    assert current_customers[promoted_id]["CustomerID"] == valid_payload["CustomerID"]
    assert current_customers[promoted_id]["submission_id"] == submission_id


def test_basic_dashboard_scoring_flow(
    metadata_dict,
    mock_preprocessor_array,
    mock_rf_model,
    mock_xgb_model,
    mock_lgbm_model,
):
    """Verify end-to-end scoring flow from raw input to probability, prediction, and risk tier."""
    print("Test 19: Dashboard scoring flow should run end-to-end from raw input to risk tier.")

    raw_input_columns = metadata_dict["raw_input_columns"]
    weights = metadata_dict["weights"]

    import pandas as pd

    model_input_df = pd.DataFrame(
        [
            {
                "CustomerID": 1001,
                "Tenure": 12,
                "PreferredLoginDevice": "Mobile Phone",
                "CityTier": 1,
                "WarehouseToHome": 15,
                "PreferredPaymentMode": "Debit Card",
                "Gender": "Male",
                "HourSpendOnApp": 3,
                "NumberOfDeviceRegistered": 2,
                "PreferedOrderCat": "Mobile Phone",
                "SatisfactionScore": 4,
                "MaritalStatus": "Single",
                "NumberOfAddress": 2,
                "Complain": 0,
                "OrderAmountHikeFromlastYear": 12,
                "CouponUsed": 1,
                "OrderCount": 3,
                "DaySinceLastOrder": 5,
                "CashbackAmount": 120.0,
                "Churn": 0,
                "source": "client_submission",
                "submission_id": "s1",
                "submitted_at": "t1",
            }
        ]
    )

    X_raw, y = prepare_model_input(model_input_df, raw_input_columns)
    X_proc = preprocess_for_ensemble(X_raw, mock_preprocessor_array, metadata_dict["selected_features"])
    probs = predict_weighted_ensemble(
        X_proc,
        mock_rf_model,
        mock_xgb_model,
        mock_lgbm_model,
        weights["w_rf"],
        weights["w_xgb"],
        weights["w_lgbm"],
    )

    preds = (probs >= 0.5).astype(int)
    tiers = [risk_tier_from_proba(p) for p in probs]

    assert len(probs) == len(X_raw)
    assert len(preds) == len(X_raw)
    assert len(tiers) == len(X_raw)
    assert set(preds.tolist()).issubset({0, 1})


def test_invalid_submission_does_not_promote(fake_db, fake_firestore, sample_schema, valid_payload):
    """Verify invalid submissions are logged but never promoted into current_customers."""
    print("Test 20: Invalid submission should be logged but not promoted.")
    bad_payload = dict(valid_payload)
    bad_payload["PreferredLoginDevice"] = ""

    errors = validate_submission(bad_payload, sample_schema)
    assert errors != []

    submission_id, promoted_id = write_to_firestore(
        db=fake_db,
        firestore_module=fake_firestore,
        payload=bad_payload,
        errors=errors,
        source_value="integration_test",
    )

    submissions = fake_db.collection("customer_submissions").documents
    current_customers = fake_db.collection("current_customers").documents

    assert submission_id is not None
    assert promoted_id is None
    assert submission_id in submissions
    assert len(current_customers) == 2