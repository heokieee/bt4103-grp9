import pandas as pd
from app_core import (
    to_binary_series,
    pick_customer_id_column,
    risk_tier_from_proba,
    simple_action_recommendations,
    preprocess_for_ensemble,
    predict_weighted_ensemble,
    prepare_model_input,
)


def test_to_binary_series_maps_common_values():
    """Verify common binary-like values are normalized into 0/1."""
    print("Test 9: Binary-like values should map correctly to 0/1.")
    s = pd.Series(["Yes", "No", "1", "0", True, False, "unknown"])
    out = to_binary_series(s)
    assert out.tolist() == [1, 0, 1, 0, 1, 0, 0]


def test_pick_customer_id_column_prefers_default_names():
    """Verify helper identifies common customer ID column names."""
    print("Test 10: Customer ID helper should detect default ID-style columns.")
    df = pd.DataFrame({"customer_id": [1, 2], "other": [3, 4]})
    assert pick_customer_id_column(df) == "customer_id"


def test_risk_tier_from_proba_buckets_correctly():
    """Verify churn probabilities map into the expected risk buckets."""
    print("Test 11: Probabilities should map into Low/Medium/High/Critical tiers.")
    assert risk_tier_from_proba(0.10) == "Low"
    assert risk_tier_from_proba(0.40) == "Medium"
    assert risk_tier_from_proba(0.70) == "High"
    assert risk_tier_from_proba(0.90) == "Critical"


def test_simple_action_recommendations_returns_expected_flags():
    """Verify recommendation logic reacts to key churn-risk signals."""
    print("Test 12: Action recommendations should reflect churn-risk triggers.")
    row = {
        "Complain": 1,
        "SatisfactionScore": 2,
        "CashbackAmount": 0,
        "CouponUsed": 0,
        "DaySinceLastOrder": 30,
        "Tenure": 2,
    }
    actions = simple_action_recommendations(row)

    assert any("complaint" in a.lower() for a in actions)
    assert any("satisfaction" in a.lower() for a in actions)
    assert any("cashback" in a.lower() or "coupon" in a.lower() for a in actions)


def test_preprocess_for_ensemble_assigns_selected_feature_names(mock_preprocessor_array):
    """Verify numpy-array preprocessing output is converted into a dataframe with selected feature names."""
    print("Test 13: Array preprocessing output should become a named feature dataframe.")
    raw = pd.DataFrame({"A": [1], "B": [2]})
    selected_features = ["f1", "f2", "f3"]

    out = preprocess_for_ensemble(raw, mock_preprocessor_array, selected_features)

    assert list(out.columns) == selected_features
    assert out.shape == (1, 3)


def test_preprocess_for_ensemble_keeps_dataframe_output(mock_preprocessor_df):
    """Verify dataframe preprocessing output is preserved without renaming."""
    print("Test 14: DataFrame preprocessing output should preserve column names.")
    raw = pd.DataFrame({"A": [1], "B": [2]})

    out = preprocess_for_ensemble(raw, mock_preprocessor_df, ["ignored"])

    assert list(out.columns) == ["p1", "p2"]
    assert out.shape == (1, 2)


def test_predict_weighted_ensemble_returns_weighted_average(
    mock_rf_model, mock_xgb_model, mock_lgbm_model
):
    """Verify ensemble prediction uses the configured weighted average across models."""
    print("Test 15: Weighted ensemble probability should match configured model weights.")
    X = pd.DataFrame({"f1": [0], "f2": [1]})

    out = predict_weighted_ensemble(
        X, mock_rf_model, mock_xgb_model, mock_lgbm_model, 0.2, 0.3, 0.5
    )

    expected = 0.2 * 0.2 + 0.3 * 0.6 + 0.5 * 0.8
    assert abs(out[0] - expected) < 1e-9


def test_prepare_model_input_drops_metadata_and_extracts_target(sample_raw_df):
    """Verify model input preparation extracts binary target, removes metadata, and keeps required raw inputs."""
    print("Test 16: Model input preparation should drop metadata and isolate target/features.")
    raw_input_columns = [
        "CustomerID",
        "Tenure",
        "PreferredLoginDevice",
        "CityTier",
        "Complain",
    ]
    X, y = prepare_model_input(sample_raw_df, raw_input_columns)

    assert y.tolist() == [0, 1]
    assert "submission_id" not in X.columns
    assert "submitted_at" not in X.columns
    assert set(raw_input_columns).issubset(set(X.columns))


def test_prepare_model_input_raises_for_missing_columns(sample_raw_df):
    """Verify model input preparation raises an error when required columns are missing."""
    print("Test 17: Missing required model columns should raise an error.")
    raw_input_columns = [
        "CustomerID",
        "Tenure",
        "PreferredLoginDevice",
        "CityTier",
        "Complain",
    ]
    broken = sample_raw_df.drop(columns=["Tenure"])

    try:
        prepare_model_input(broken, raw_input_columns)
        assert False, "Expected ValueError for missing required columns."
    except ValueError as e:
        assert "missing" in str(e).lower()