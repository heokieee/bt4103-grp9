from frontend_core import (
    load_dataset_schema,
    get_next_customer_id,
    validate_submission,
    write_to_firestore,
)


def test_load_dataset_schema_returns_expected_keys(real_dataset_csv):
    """Verify schema loader reads the CSV and returns expected structural keys."""
    print("Test 1: Loading dataset schema and checking expected keys/column groups.")
    schema = load_dataset_schema(real_dataset_csv)

    assert "columns" in schema
    assert "numeric_cols" in schema
    assert "categorical_cols" in schema
    assert "categorical_values" in schema
    assert "numeric_stats" in schema
    assert "CustomerID" in schema["columns"]
    assert "PreferredLoginDevice" in schema["categorical_cols"]


def test_get_next_customer_id_uses_highest_existing_id(fake_db, fake_firestore):
    """Verify next CustomerID is computed from the highest existing current_customers record."""
    print("Test 2: Checking next CustomerID generation from existing Firestore records.")
    next_id = get_next_customer_id(fake_db, fake_firestore)
    assert next_id == 1006


def test_validate_submission_accepts_valid_payload(sample_schema, valid_payload):
    """Verify a valid payload passes validation with no errors."""
    print("Test 3: Valid payload should pass validation with zero errors.")
    errors = validate_submission(valid_payload, sample_schema)
    assert errors == []


def test_validate_submission_flags_missing_required_value(sample_schema, valid_payload):
    """Verify validation catches a missing required categorical field."""
    print("Test 4: Missing required field should be flagged.")
    payload = dict(valid_payload)
    payload["PreferredLoginDevice"] = ""

    errors = validate_submission(payload, sample_schema)
    assert any("PreferredLoginDevice" in err for err in errors)


def test_validate_submission_flags_invalid_category(sample_schema, valid_payload):
    """Verify validation rejects categories outside the allowed schema values."""
    print("Test 5: Invalid category should be rejected.")
    payload = dict(valid_payload)
    payload["PreferredLoginDevice"] = "Tablet"

    errors = validate_submission(payload, sample_schema)
    assert any("PreferredLoginDevice" in err for err in errors)


def test_validate_submission_flags_range_and_integer_errors(sample_schema, valid_payload):
    """Verify validation catches both out-of-range numeric values and non-integer integer fields."""
    print("Test 6: Out-of-range and non-integer numeric values should be flagged.")
    payload = dict(valid_payload)
    payload["Tenure"] = 999
    payload["CityTier"] = 2.5

    errors = validate_submission(payload, sample_schema)
    assert any("Tenure" in err for err in errors)
    assert any("CityTier" in err for err in errors)


def test_write_to_firestore_invalid_submission_only_hits_submissions(fake_db, fake_firestore, valid_payload):
    """Verify invalid submissions are logged in customer_submissions only and not promoted."""
    print("Test 7: Invalid submission should stay in customer_submissions only.")
    errors = ["PreferredLoginDevice: value is required"]

    submission_id, promoted_id = write_to_firestore(
        db=fake_db,
        firestore_module=fake_firestore,
        payload=valid_payload,
        errors=errors,
        source_value="streamlit_form",
    )

    submissions = fake_db.collection("customer_submissions").documents
    current_customers = fake_db.collection("current_customers").documents

    assert submission_id is not None
    assert promoted_id is None
    assert len(submissions) == 1
    assert len(current_customers) == 2  # only seeded docs remain

    stored = submissions[submission_id]
    assert stored["validation_status"] == "invalid"
    assert stored["validation_errors"] == errors


def test_write_to_firestore_valid_submission_promotes_to_current(fake_db, fake_firestore, valid_payload):
    """Verify valid submissions are logged and promoted into current_customers."""
    print("Test 8: Valid submission should be logged and promoted to current_customers.")
    submission_id, promoted_id = write_to_firestore(
        db=fake_db,
        firestore_module=fake_firestore,
        payload=valid_payload,
        errors=[],
        source_value="streamlit_form",
    )

    submissions = fake_db.collection("customer_submissions").documents
    current_customers = fake_db.collection("current_customers").documents

    assert submission_id is not None
    assert promoted_id == submission_id
    assert len(submissions) == 1
    assert len(current_customers) == 3

    promoted = current_customers[promoted_id]
    assert promoted["CustomerID"] == valid_payload["CustomerID"]
    assert promoted["submission_id"] == submission_id
    assert promoted["source"] == "streamlit_form"
    assert "created_at_utc" in promoted