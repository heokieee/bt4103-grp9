import time
import pandas as pd
import numpy as np
import pytest
from app_core import (
    to_binary_series,
    pick_customer_id_column,
    risk_tier_from_proba,
    simple_action_recommendations,
    preprocess_for_ensemble,
    predict_weighted_ensemble,
    prepare_model_input,
)
 
 
# ===========================================================================
# ORIGINAL TESTS (9–17)
# ===========================================================================
 
def test_to_binary_series_maps_common_values():
    """Verify common binary-like values are normalized into 0/1."""
    print("Test 9: Binary-like values should map correctly to 0/1.")
    s = pd.Series(["Yes", "No", "1", "0", True, False, "unknown"])
    start = time.perf_counter()
    out = to_binary_series(s)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert out.tolist() == [1, 0, 1, 0, 1, 0, 0]
    print(f"  RESULT: Mapped 7 mixed values → {out.tolist()} in {elapsed_ms:.3f} ms")
 
 
def test_pick_customer_id_column_prefers_default_names():
    """Verify helper identifies common customer ID column names."""
    print("Test 10: Customer ID helper should detect default ID-style columns.")
    df = pd.DataFrame({"customer_id": [1, 2], "other": [3, 4]})
    start = time.perf_counter()
    result = pick_customer_id_column(df)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert result == "customer_id"
    print(f"  RESULT: Detected ID column '{result}' from 2-column DataFrame in {elapsed_ms:.3f} ms")
 
 
def test_risk_tier_from_proba_buckets_correctly():
    """Verify churn probabilities map into the expected risk buckets."""
    print("Test 11: Probabilities should map into Low/Medium/High/Critical tiers.")
    cases = [(0.10, "Low"), (0.40, "Medium"), (0.70, "High"), (0.90, "Critical")]
    start = time.perf_counter()
    results = [(p, risk_tier_from_proba(p)) for p, _ in cases]
    elapsed_ms = (time.perf_counter() - start) * 1000
    for (p, expected), (_, got) in zip(cases, results):
        assert got == expected
    summary = ", ".join(f"{p:.0%}→{t}" for p, t in results)
    print(f"  RESULT: Risk tier assignments: {summary} ({elapsed_ms:.3f} ms)")
 
 
def test_simple_action_recommendations_returns_expected_flags():
    """Verify recommendation logic reacts to key churn-risk signals."""
    print("Test 12: Action recommendations should reflect churn-risk triggers.")
    row = {
        "Complain": 1, "SatisfactionScore": 2, "CashbackAmount": 0,
        "CouponUsed": 0, "DaySinceLastOrder": 30, "Tenure": 2,
    }
    start = time.perf_counter()
    actions = simple_action_recommendations(row)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert any("complaint" in a.lower() for a in actions)
    assert any("satisfaction" in a.lower() for a in actions)
    assert any("cashback" in a.lower() or "coupon" in a.lower() for a in actions)
    print(f"  RESULT: Generated {len(actions)} action recommendations for high-risk customer in {elapsed_ms:.3f} ms")
 
 
def test_preprocess_for_ensemble_assigns_selected_feature_names(mock_preprocessor_array):
    """Verify numpy-array preprocessing output is converted into a dataframe with selected feature names."""
    print("Test 13: Array preprocessing output should become a named feature dataframe.")
    raw = pd.DataFrame({"A": [1], "B": [2]})
    selected_features = ["f1", "f2", "f3"]
    start = time.perf_counter()
    out = preprocess_for_ensemble(raw, mock_preprocessor_array, selected_features)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert list(out.columns) == selected_features
    assert out.shape == (1, 3)
    print(f"  RESULT: Preprocessed 1 customer row → shape {out.shape}, columns {list(out.columns)} in {elapsed_ms:.3f} ms")
 
 
def test_preprocess_for_ensemble_keeps_dataframe_output(mock_preprocessor_df):
    """Verify dataframe preprocessing output is preserved without renaming."""
    print("Test 14: DataFrame preprocessing output should preserve column names.")
    raw = pd.DataFrame({"A": [1], "B": [2]})
    start = time.perf_counter()
    out = preprocess_for_ensemble(raw, mock_preprocessor_df, ["ignored"])
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert list(out.columns) == ["p1", "p2"]
    assert out.shape == (1, 2)
    print(f"  RESULT: Pipeline preserved DataFrame output with columns {list(out.columns)} in {elapsed_ms:.3f} ms")
 
 
def test_predict_weighted_ensemble_returns_weighted_average(
    mock_rf_model, mock_xgb_model, mock_lgbm_model
):
    """Verify ensemble prediction uses the configured weighted average across models."""
    print("Test 15: Weighted ensemble probability should match configured model weights.")
    X = pd.DataFrame({"f1": [0], "f2": [1]})
    start = time.perf_counter()
    out = predict_weighted_ensemble(X, mock_rf_model, mock_xgb_model, mock_lgbm_model, 0.2, 0.3, 0.5)
    elapsed_ms = (time.perf_counter() - start) * 1000
    expected = 0.2 * 0.2 + 0.3 * 0.6 + 0.5 * 0.8
    assert abs(out[0] - expected) < 1e-9
    print(f"  RESULT: Churn probability = {out[0]:.4f} (RF×0.2 + XGB×0.3 + LGBM×0.5) computed in {elapsed_ms:.3f} ms")
 
 
def test_prepare_model_input_drops_metadata_and_extracts_target(sample_raw_df):
    """Verify model input preparation extracts binary target, removes metadata, and keeps required raw inputs."""
    print("Test 16: Model input preparation should drop metadata and isolate target/features.")
    raw_input_columns = ["CustomerID", "Tenure", "PreferredLoginDevice", "CityTier", "Complain"]
    start = time.perf_counter()
    X, y = prepare_model_input(sample_raw_df, raw_input_columns)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert y.tolist() == [0, 1]
    assert "submission_id" not in X.columns
    assert "submitted_at" not in X.columns
    assert set(raw_input_columns).issubset(set(X.columns))
    print(f"  RESULT: Prepared {len(X)} rows × {len(X.columns)} features, target labels {y.tolist()}, metadata stripped in {elapsed_ms:.3f} ms")
 
 
def test_prepare_model_input_raises_for_missing_columns(sample_raw_df):
    """Verify model input preparation raises an error when required columns are missing."""
    print("Test 17: Missing required model columns should raise an error.")
    raw_input_columns = ["CustomerID", "Tenure", "PreferredLoginDevice", "CityTier", "Complain"]
    broken = sample_raw_df.drop(columns=["Tenure"])
    start = time.perf_counter()
    try:
        prepare_model_input(broken, raw_input_columns)
        assert False, "Expected ValueError for missing required columns."
    except ValueError as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert "missing" in str(e).lower()
        print(f"  RESULT: ValueError correctly raised for missing 'Tenure' column in {elapsed_ms:.3f} ms — dashboard won't silently score incomplete data")
 
 
# ===========================================================================
# EXTENDED TESTS (21–57)
# ===========================================================================
 
# --- risk_tier_from_proba — boundary values ---
 
@pytest.mark.parametrize("proba,expected", [
    (0.0,   "Low"),
    (0.001, "Low"),
    (0.249, "Low"),
    (0.25,  "Medium"),
    (0.499, "Medium"),
    (0.50,  "High"),
    (0.749, "High"),
    (0.75,  "Critical"),
    (0.999, "Critical"),
    (1.0,   "Critical"),
])
def test_risk_tier_boundary_values(proba, expected):
    """Verify every threshold boundary maps to the correct tier."""
    start = time.perf_counter()
    result = risk_tier_from_proba(proba)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert result == expected
    print(f"  RESULT: Churn probability {proba:.3f} → tier '{result}' (boundary check passed in {elapsed_ms:.3f} ms)")
 
 
def test_risk_tier_exhaustive_no_unexpected_values():
    """Every probability in [0,1] must map to one of the four valid tiers."""
    print("Test 22: All probabilities in [0,1] must map to a valid tier.")
    valid = {"Low", "Medium", "High", "Critical"}
    probas = np.linspace(0.0, 1.0, 201)
    start = time.perf_counter()
    tiers = [risk_tier_from_proba(float(p)) for p in probas]
    elapsed_ms = (time.perf_counter() - start) * 1000
    for p, tier in zip(probas, tiers):
        assert tier in valid, f"Unexpected tier '{tier}' for p={p:.4f}"
    counts = {t: tiers.count(t) for t in valid}
    print(f"  RESULT: 201 probabilities classified in {elapsed_ms:.2f} ms → Low:{counts['Low']}, Medium:{counts['Medium']}, High:{counts['High']}, Critical:{counts['Critical']}")
 
 
# --- to_binary_series — edge cases ---
 
def test_to_binary_series_numeric_int_passthrough():
    """Integer 0/1 series should pass through unchanged."""
    print("Test 23: Integer 0/1 series passthrough.")
    s = pd.Series([0, 1, 0, 1])
    start = time.perf_counter()
    out = to_binary_series(s)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert out.tolist() == [0, 1, 0, 1]
    print(f"  RESULT: Integer series passed through unchanged → {out.tolist()} in {elapsed_ms:.3f} ms")
 
 
def test_to_binary_series_bool_values():
    """Boolean series should map True→1, False→0."""
    print("Test 24: Boolean series True/False → 1/0.")
    s = pd.Series([True, False, True])
    start = time.perf_counter()
    out = to_binary_series(s)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert out.tolist() == [1, 0, 1]
    print(f"  RESULT: Boolean values mapped → {out.tolist()} in {elapsed_ms:.3f} ms")
 
 
def test_to_binary_series_nan_defaults_to_zero():
    """NaN values must become 0 without raising."""
    print("Test 25: NaN in binary series defaults to 0.")
    s = pd.Series([np.nan, 1, 0])
    start = time.perf_counter()
    out = to_binary_series(s)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert out.iloc[0] == 0
    print(f"  RESULT: NaN safely defaulted to 0, series → {out.tolist()} in {elapsed_ms:.3f} ms")
 
 
def test_to_binary_series_mixed_types():
    """Mixed string/int/bool input must all map correctly."""
    print("Test 26: Mixed string/int/bool values should map to 1/0.")
    s = pd.Series(["1", 0, "Yes", "No", True])
    start = time.perf_counter()
    out = to_binary_series(s)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert out.tolist() == [1, 0, 1, 0, 1]
    print(f"  RESULT: Mixed-type input normalised → {out.tolist()} in {elapsed_ms:.3f} ms")
 
 
def test_to_binary_series_all_unknown_defaults_to_zero():
    """Unrecognised values must all fall back to 0."""
    print("Test 27: Unknown values should default to 0.")
    s = pd.Series(["maybe", "dunno", None])
    start = time.perf_counter()
    out = to_binary_series(s)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert all(v == 0 for v in out)
    print(f"  RESULT: 3 unrecognised values safely defaulted to 0 in {elapsed_ms:.3f} ms")
 
 
def test_to_binary_series_empty_series_no_error():
    """Empty series must return an empty series without raising."""
    print("Test 28: Empty series should return empty without error.")
    start = time.perf_counter()
    out = to_binary_series(pd.Series([], dtype=object))
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert len(out) == 0
    print(f"  RESULT: Empty series handled gracefully, returned length {len(out)} in {elapsed_ms:.3f} ms")
 
 
def test_to_binary_series_large_all_yes():
    """10 000 'Yes' values must all map to 1."""
    print("Test 29: Large all-Yes series should all map to 1.")
    s = pd.Series(["Yes"] * 10_000)
    start = time.perf_counter()
    out = to_binary_series(s)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert (out == 1).all()
    print(f"  RESULT: 10,000 'Yes' values all mapped to 1 in {elapsed_ms:.2f} ms")
 
 
# --- simple_action_recommendations ---
 
def _healthy_row() -> dict:
    return {
        "Complain": 0, "SatisfactionScore": 5, "CashbackAmount": 200,
        "CouponUsed": 5, "DaySinceLastOrder": 2, "Tenure": 24,
    }
 
 
def test_action_no_triggers_returns_default():
    """Healthy customer must return only the default no-action message."""
    print("Test 30: Healthy row should return default no-action message.")
    start = time.perf_counter()
    actions = simple_action_recommendations(_healthy_row())
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert len(actions) == 1
    assert "no major" in actions[0].lower()
    print(f"  RESULT: Healthy customer → 1 action: '{actions[0]}' ({elapsed_ms:.3f} ms)")
 
 
def test_action_all_six_triggers_fire():
    """High-risk row with every signal set must fire all 6 recommendations."""
    print("Test 31: All-bad row should trigger all 6 action recommendations.")
    row = {
        "Complain": 1, "SatisfactionScore": 1, "CashbackAmount": 10,
        "CouponUsed": 0, "DaySinceLastOrder": 30, "Tenure": 1,
    }
    start = time.perf_counter()
    actions = simple_action_recommendations(row)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert len(actions) == 6
    print(f"  RESULT: High-risk customer triggered all 6 dashboard actions in {elapsed_ms:.3f} ms")
 
 
def test_action_complain_zero_no_complaint_action():
    """Complain=0 must not trigger the complaint recommendation."""
    print("Test 32: Complain=0 should not trigger complaint action.")
    row = dict(_healthy_row()); row["Complain"] = 0
    start = time.perf_counter()
    actions = simple_action_recommendations(row)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert not any("complaint" in a.lower() for a in actions)
    print(f"  RESULT: No complaint flag → complaint action correctly suppressed in {elapsed_ms:.3f} ms")
 
 
def test_action_complain_one_triggers():
    """Complain=1 must trigger the complaint follow-up recommendation."""
    print("Test 33: Complain=1 should trigger complaint action.")
    row = dict(_healthy_row()); row["Complain"] = 1
    start = time.perf_counter()
    actions = simple_action_recommendations(row)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert any("complaint" in a.lower() for a in actions)
    print(f"  RESULT: Complaint flag set → complaint follow-up action shown on dashboard in {elapsed_ms:.3f} ms")
 
 
def test_action_tenure_two_triggers_onboarding():
    """Tenure exactly 2 must trigger the onboarding recommendation."""
    print("Test 34: Tenure=2 should trigger onboarding action.")
    row = dict(_healthy_row()); row["Tenure"] = 2
    start = time.perf_counter()
    actions = simple_action_recommendations(row)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert any("onboarding" in a.lower() for a in actions)
    print(f"  RESULT: Tenure=2 months → onboarding action correctly displayed in {elapsed_ms:.3f} ms")
 
 
def test_action_tenure_three_no_onboarding():
    """Tenure=3 must NOT trigger onboarding (condition is <= 2)."""
    print("Test 35: Tenure=3 should not trigger onboarding action.")
    row = dict(_healthy_row()); row["Tenure"] = 3
    start = time.perf_counter()
    actions = simple_action_recommendations(row)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert not any("onboarding" in a.lower() for a in actions)
    print(f"  RESULT: Tenure=3 months → onboarding action correctly hidden in {elapsed_ms:.3f} ms")
 
 
def test_action_days_since_order_ten_no_winback():
    """DaySinceLastOrder exactly 10 must NOT trigger win-back (condition is > 10)."""
    print("Test 36: DaySinceLastOrder=10 should not trigger win-back.")
    row = dict(_healthy_row()); row["DaySinceLastOrder"] = 10
    start = time.perf_counter()
    actions = simple_action_recommendations(row)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert not any("win-back" in a.lower() for a in actions)
    print(f"  RESULT: 10 days since last order → win-back campaign correctly not triggered in {elapsed_ms:.3f} ms")
 
 
def test_action_days_since_order_eleven_triggers_winback():
    """DaySinceLastOrder=11 must trigger win-back campaign."""
    print("Test 37: DaySinceLastOrder=11 should trigger win-back action.")
    row = dict(_healthy_row()); row["DaySinceLastOrder"] = 11
    start = time.perf_counter()
    actions = simple_action_recommendations(row)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert any("win-back" in a.lower() for a in actions)
    print(f"  RESULT: 11 days since last order → win-back campaign action shown on dashboard in {elapsed_ms:.3f} ms")
 
 
def test_action_missing_keys_safe_defaults_no_crash():
    """Empty dict must not raise — defaults prevent all triggers."""
    print("Test 38: Empty row dict should not crash and return default message.")
    start = time.perf_counter()
    actions = simple_action_recommendations({})
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert len(actions) == 1
    print(f"  RESULT: Missing customer data handled safely → '{actions[0]}' in {elapsed_ms:.3f} ms")
 
 
# --- pick_customer_id_column ---
 
def test_pick_id_prefers_CustomerID_exact():
    """CustomerID column must be preferred over other ID candidates."""
    print("Test 39: CustomerID exact match should be preferred.")
    df = pd.DataFrame({"CustomerID": [1], "customer_id": [2]})
    start = time.perf_counter()
    result = pick_customer_id_column(df)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert result == "CustomerID"
    print(f"  RESULT: Dashboard ID column resolved to '{result}' in {elapsed_ms:.3f} ms")
 
 
def test_pick_id_falls_back_to_lower_customer_id():
    """customer_id (lowercase) must be found when CustomerID is absent."""
    print("Test 40: Lowercase customer_id should be found as fallback.")
    df = pd.DataFrame({"customer_id": [1], "value": [2]})
    start = time.perf_counter()
    result = pick_customer_id_column(df)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert result == "customer_id"
    print(f"  RESULT: Fallback ID column detected as '{result}' in {elapsed_ms:.3f} ms")
 
 
def test_pick_id_fuzzy_match_customer_and_id_in_name():
    """Column containing both 'customer' and 'id' must be matched."""
    print("Test 41: Fuzzy match on column containing 'customer' and 'id'.")
    df = pd.DataFrame({"my_customer_identifier": [1], "other": [2]})
    start = time.perf_counter()
    result = pick_customer_id_column(df)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert result == "my_customer_identifier"
    print(f"  RESULT: Fuzzy-matched ID column '{result}' for customer lookup in {elapsed_ms:.3f} ms")
 
 
def test_pick_id_returns_none_when_no_match():
    """Should return None when no ID-like column exists."""
    print("Test 42: No matching ID column should return None.")
    df = pd.DataFrame({"revenue": [1], "name": [2]})
    start = time.perf_counter()
    result = pick_customer_id_column(df)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert result is None
    print(f"  RESULT: No ID column found → customer lookup tab will show warning in {elapsed_ms:.3f} ms")
 
 
def test_pick_id_plain_id_column_matches():
    """Plain 'id' column must match (it is in the explicit candidates list)."""
    print("Test 43: Plain 'id' column should match.")
    df = pd.DataFrame({"id": [1], "name": [2]})
    start = time.perf_counter()
    result = pick_customer_id_column(df)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert result == "id"
    print(f"  RESULT: Plain 'id' column resolved for customer lookup in {elapsed_ms:.3f} ms")
 
 
# --- preprocess_for_ensemble ---
 
def test_preprocess_numpy_output_gets_feature_names(mock_preprocessor_array):
    """Numpy array output must be renamed to selected_features."""
    print("Test 44: Numpy output should be renamed to selected feature names.")
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    start = time.perf_counter()
    out = preprocess_for_ensemble(df, mock_preprocessor_array, ["f1", "f2", "f3"])
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert list(out.columns) == ["f1", "f2", "f3"]
    print(f"  RESULT: Preprocessing pipeline output renamed to {list(out.columns)} ready for scoring in {elapsed_ms:.3f} ms")
 
 
def test_preprocess_dataframe_output_preserves_columns(mock_preprocessor_df):
    """DataFrame pipeline output must keep its own column names."""
    print("Test 45: DataFrame output should preserve original column names.")
    df = pd.DataFrame({"A": [1], "B": [2]})
    start = time.perf_counter()
    out = preprocess_for_ensemble(df, mock_preprocessor_df, ["ignored"])
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert list(out.columns) == ["p1", "p2"]
    print(f"  RESULT: Pipeline preserved column names {list(out.columns)} in {elapsed_ms:.3f} ms")
 
 
def test_preprocess_row_count_matches_input(mock_preprocessor_array):
    """Output row count must equal input row count."""
    print("Test 46: Output row count should match input row count.")
    df = pd.DataFrame({"A": range(10), "B": range(10)})
    start = time.perf_counter()
    out = preprocess_for_ensemble(df, mock_preprocessor_array, ["f1", "f2", "f3"])
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert out.shape[0] == 10
    print(f"  RESULT: 10 customer rows preprocessed → {out.shape[0]} rows output (no rows dropped) in {elapsed_ms:.3f} ms")
 
 
def test_preprocess_always_returns_dataframe(mock_preprocessor_array):
    """preprocess_for_ensemble must always return a pd.DataFrame."""
    print("Test 47: Output type should always be a pd.DataFrame.")
    df = pd.DataFrame({"A": [1, 2, 3]})
    start = time.perf_counter()
    out = preprocess_for_ensemble(df, mock_preprocessor_array, ["f1", "f2", "f3"])
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert isinstance(out, pd.DataFrame)
    print(f"  RESULT: Preprocessing always returns DataFrame (shape {out.shape}) in {elapsed_ms:.3f} ms")
 
 
def test_preprocess_feature_mismatch_keeps_int_columns(mock_preprocessor_array):
    """When selected_features length != transformed cols, columns stay as ints."""
    print("Test 48: Feature count mismatch should keep integer column indices.")
    df = pd.DataFrame({"A": [1], "B": [2]})
    start = time.perf_counter()
    out = preprocess_for_ensemble(df, mock_preprocessor_array, ["f1", "f2"])
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert all(isinstance(c, (int, np.integer)) for c in out.columns)
    print(f"  RESULT: Feature count mismatch handled safely, columns kept as {list(out.columns)} in {elapsed_ms:.3f} ms")
 
 
# --- predict_weighted_ensemble ---
 
def test_predict_ensemble_correct_weighted_average(mock_rf_model, mock_xgb_model, mock_lgbm_model):
    """Ensemble output must exactly match the configured weighted average."""
    print("Test 49: Weighted ensemble output should match manual calculation.")
    X = pd.DataFrame({"f1": [0], "f2": [1]})
    start = time.perf_counter()
    out = predict_weighted_ensemble(X, mock_rf_model, mock_xgb_model, mock_lgbm_model, 0.2, 0.3, 0.5)
    elapsed_ms = (time.perf_counter() - start) * 1000
    expected = 0.2 * 0.2 + 0.3 * 0.6 + 0.5 * 0.8
    assert abs(out[0] - expected) < 1e-9
    print(f"  RESULT: Ensemble churn probability = {out[0]:.4f} (expected {expected:.4f}) scored in {elapsed_ms:.3f} ms")
 
 
def test_predict_ensemble_equal_weights(mock_rf_model, mock_xgb_model, mock_lgbm_model):
    """Equal weights (1/3 each) must produce the unweighted mean."""
    print("Test 50: Equal weights should produce unweighted mean of model probabilities.")
    X = pd.DataFrame({"f1": [0]})
    w = 1 / 3
    start = time.perf_counter()
    out = predict_weighted_ensemble(X, mock_rf_model, mock_xgb_model, mock_lgbm_model, w, w, w)
    elapsed_ms = (time.perf_counter() - start) * 1000
    expected = w * 0.2 + w * 0.6 + w * 0.8
    assert abs(out[0] - expected) < 1e-9
    print(f"  RESULT: Equal-weight ensemble probability = {out[0]:.4f} in {elapsed_ms:.3f} ms")
 
 
def test_predict_ensemble_output_in_zero_one_range(mock_rf_model, mock_xgb_model, mock_lgbm_model):
    """All ensemble probabilities must be within [0.0, 1.0]."""
    print("Test 51: All ensemble probabilities should be in [0, 1].")
    X = pd.DataFrame({"f1": range(50)})
    start = time.perf_counter()
    out = predict_weighted_ensemble(X, mock_rf_model, mock_xgb_model, mock_lgbm_model, 0.2, 0.3, 0.5)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert (out >= 0.0).all() and (out <= 1.0).all()
    print(f"  RESULT: 50 churn probabilities all in [0,1], range [{out.min():.3f}, {out.max():.3f}] in {elapsed_ms:.3f} ms")
 
 
def test_predict_ensemble_length_matches_input(mock_rf_model, mock_xgb_model, mock_lgbm_model):
    """Output length must equal the number of input rows."""
    print("Test 52: Output length should match number of input rows.")
    X = pd.DataFrame({"f": range(37)})
    start = time.perf_counter()
    out = predict_weighted_ensemble(X, mock_rf_model, mock_xgb_model, mock_lgbm_model, 0.33, 0.33, 0.34)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert len(out) == 37
    print(f"  RESULT: 37 customers scored → 37 churn probabilities returned in {elapsed_ms:.3f} ms")
 
 
# --- prepare_model_input — edge cases ---
 
def test_prepare_no_churn_column_returns_none_y():
    """DataFrame without Churn column must return y=None."""
    print("Test 53: Missing Churn column should return y=None.")
    df = pd.DataFrame({"CustomerID": [1], "Tenure": [5], "PreferredLoginDevice": ["Mobile Phone"]})
    start = time.perf_counter()
    X, y = prepare_model_input(df, ["CustomerID", "Tenure", "PreferredLoginDevice"])
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert y is None
    print(f"  RESULT: New customer (no Churn label) → y=None, {len(X.columns)} features extracted in {elapsed_ms:.3f} ms")
 
 
def test_prepare_metadata_columns_stripped():
    """All metadata columns must be excluded from X."""
    print("Test 54: Metadata columns should be excluded from model input X.")
    df = pd.DataFrame({
        "CustomerID": [1], "Tenure": [5], "Churn": [0],
        "source": ["test"], "submission_id": ["s1"],
        "submitted_at": ["t1"], "validation_status": ["valid"],
    })
    start = time.perf_counter()
    X, _ = prepare_model_input(df, ["CustomerID", "Tenure"])
    elapsed_ms = (time.perf_counter() - start) * 1000
    for col in ["source", "submission_id", "submitted_at", "validation_status"]:
        assert col not in X.columns
    print(f"  RESULT: 4 Firestore metadata columns stripped, model receives {list(X.columns)} in {elapsed_ms:.3f} ms")
 
 
def test_prepare_churn_yes_no_strings_encoded():
    """Churn='Yes'/'No' strings must be converted to 1/0."""
    print("Test 55: Yes/No Churn strings should encode to 1/0.")
    df = pd.DataFrame({"CustomerID": [1, 2], "Tenure": [5, 10], "Churn": ["Yes", "No"]})
    start = time.perf_counter()
    _, y = prepare_model_input(df, ["CustomerID", "Tenure"])
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert y.tolist() == [1, 0]
    print(f"  RESULT: 'Yes'/'No' Churn labels encoded to {y.tolist()} in {elapsed_ms:.3f} ms")
 
 
def test_prepare_single_row_correct_shape():
    """Single-row DataFrame must produce correctly shaped X and y."""
    print("Test 56: Single-row input should produce shape (1, n) for X.")
    df = pd.DataFrame({"CustomerID": [1], "Tenure": [5], "Churn": [1]})
    start = time.perf_counter()
    X, y = prepare_model_input(df, ["CustomerID", "Tenure"])
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert X.shape == (1, 2)
    assert y.tolist() == [1]
    print(f"  RESULT: Single customer row → X shape {X.shape}, Churn label {y.tolist()} in {elapsed_ms:.3f} ms")
 
 
def test_prepare_missing_required_columns_raises(sample_raw_df):
    """Missing required columns must raise a ValueError."""
    print("Test 57: Missing required columns should raise ValueError.")
    broken = sample_raw_df.drop(columns=["Tenure"])
    start = time.perf_counter()
    with pytest.raises(ValueError, match="(?i)missing"):
        prepare_model_input(broken, ["CustomerID", "Tenure", "PreferredLoginDevice", "CityTier", "Complain"])
    elapsed_ms = (time.perf_counter() - start) * 1000
    print(f"  RESULT: Missing 'Tenure' column caught in {elapsed_ms:.3f} ms — prevents silent bad scoring on dashboard")
