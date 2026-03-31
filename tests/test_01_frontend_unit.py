import time
import pytest
from frontend_core import (
    load_dataset_schema,
    get_next_customer_id,
    validate_submission,
    write_to_firestore,
)


# ===========================================================================
# ORIGINAL TESTS (1–8)
# ===========================================================================

def test_load_dataset_schema_returns_expected_keys(real_dataset_csv):
    """Verify schema loader reads the CSV and returns expected structural keys."""
    print("Test 1: Loading dataset schema and checking expected keys/column groups.")
    start = time.perf_counter()
    schema = load_dataset_schema(real_dataset_csv)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert "columns" in schema
    assert "numeric_cols" in schema
    assert "categorical_cols" in schema
    assert "categorical_values" in schema
    assert "numeric_stats" in schema
    assert "CustomerID" in schema["columns"]
    assert "PreferredLoginDevice" in schema["categorical_cols"]
    print(f"  RESULT: Schema loaded in {elapsed_ms:.2f} ms — {len(schema['columns'])} columns ({len(schema['numeric_cols'])} numeric, {len(schema['categorical_cols'])} categorical)")


def test_get_next_customer_id_uses_highest_existing_id(fake_db, fake_firestore):
    """Verify next CustomerID is computed from the highest existing current_customers record."""
    print("Test 2: Checking next CustomerID generation from existing Firestore records.")
    start = time.perf_counter()
    next_id = get_next_customer_id(fake_db, fake_firestore)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert next_id == 1006
    print(f"  RESULT: Next CustomerID auto-assigned as {next_id} (highest existing + 1) in {elapsed_ms:.3f} ms")


def test_validate_submission_accepts_valid_payload(sample_schema, valid_payload):
    """Verify a valid payload passes validation with no errors."""
    print("Test 3: Valid payload should pass validation with zero errors.")
    start = time.perf_counter()
    errors = validate_submission(valid_payload, sample_schema)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert errors == []
    print(f"  RESULT: Customer intake form validated in {elapsed_ms:.3f} ms — 0 errors, ready to submit")


def test_validate_submission_flags_missing_required_value(sample_schema, valid_payload):
    """Verify validation catches a missing required categorical field."""
    print("Test 4: Missing required field should be flagged.")
    payload = dict(valid_payload)
    payload["PreferredLoginDevice"] = ""
    start = time.perf_counter()
    errors = validate_submission(payload, sample_schema)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert any("PreferredLoginDevice" in err for err in errors)
    print(f"  RESULT: Blank 'PreferredLoginDevice' caught in {elapsed_ms:.3f} ms — form will show inline error to user")


def test_validate_submission_flags_invalid_category(sample_schema, valid_payload):
    """Verify validation rejects categories outside the allowed schema values."""
    print("Test 5: Invalid category should be rejected.")
    payload = dict(valid_payload)
    payload["PreferredLoginDevice"] = "Tablet"
    start = time.perf_counter()
    errors = validate_submission(payload, sample_schema)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert any("PreferredLoginDevice" in err for err in errors)
    print(f"  RESULT: 'Tablet' rejected as invalid category in {elapsed_ms:.3f} ms — form dropdown prevents bad data entering Firestore")


def test_validate_submission_flags_range_and_integer_errors(sample_schema, valid_payload):
    """Verify validation catches both out-of-range numeric values and non-integer integer fields."""
    print("Test 6: Out-of-range and non-integer numeric values should be flagged.")
    payload = dict(valid_payload)
    payload["Tenure"] = 999
    payload["CityTier"] = 2.5
    start = time.perf_counter()
    errors = validate_submission(payload, sample_schema)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert any("Tenure" in err for err in errors)
    assert any("CityTier" in err for err in errors)
    print(f"  RESULT: Tenure=999 (out of range) and CityTier=2.5 (non-integer) both caught in {elapsed_ms:.3f} ms — {len(errors)} error(s) returned")


def test_write_to_firestore_invalid_submission_only_hits_submissions(fake_db, fake_firestore, valid_payload):
    """Verify invalid submissions are logged in customer_submissions only and not promoted."""
    print("Test 7: Invalid submission should stay in customer_submissions only.")
    errors = ["PreferredLoginDevice: value is required"]
    start = time.perf_counter()
    submission_id, promoted_id = write_to_firestore(
        db=fake_db, firestore_module=fake_firestore,
        payload=valid_payload, errors=errors, source_value="streamlit_form",
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    submissions = fake_db.collection("customer_submissions").documents
    current_customers = fake_db.collection("current_customers").documents
    assert submission_id is not None
    assert promoted_id is None
    assert len(submissions) == 1
    assert len(current_customers) == 2
    stored = submissions[submission_id]
    assert stored["validation_status"] == "invalid"
    assert stored["validation_errors"] == errors
    print(f"  RESULT: Invalid record written to customer_submissions (id={submission_id}) in {elapsed_ms:.3f} ms — blocked from current_customers, current_customers still has {len(current_customers)} records")


def test_write_to_firestore_valid_submission_promotes_to_current(fake_db, fake_firestore, valid_payload):
    """Verify valid submissions are logged and promoted into current_customers."""
    print("Test 8: Valid submission should be logged and promoted to current_customers.")
    start = time.perf_counter()
    submission_id, promoted_id = write_to_firestore(
        db=fake_db, firestore_module=fake_firestore,
        payload=valid_payload, errors=[], source_value="streamlit_form",
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
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
    print(f"  RESULT: Valid customer intake written and promoted in {elapsed_ms:.3f} ms — current_customers now has {len(current_customers)} records (CustomerID={promoted['CustomerID']})")


# ===========================================================================
# EXTENDED TESTS (58–79)
# ===========================================================================

# --- validate_submission — edge cases & boundary conditions ---

def test_validate_multiple_invalid_fields_all_reported(sample_schema):
    """Each invalid field must produce its own error — no silent swallowing."""
    print("Test 58: Multiple invalid fields should each produce a separate error.")
    payload = {
        "CustomerID": 1006, "Tenure": 999, "PreferredLoginDevice": "Tablet",
        "CityTier": -1, "Complain": 0, "Churn": None,
    }
    start = time.perf_counter()
    errors = validate_submission(payload, sample_schema)
    elapsed_ms = (time.perf_counter() - start) * 1000
    joined = " ".join(errors)
    assert "Tenure" in joined
    assert "PreferredLoginDevice" in joined
    assert "CityTier" in joined
    print(f"  RESULT: {len(errors)} field errors returned in {elapsed_ms:.3f} ms — form will highlight all {len(errors)} problems at once")


def test_validate_tenure_zero_is_valid_lower_bound(sample_schema, valid_payload):
    """Tenure == 0 is the minimum and must pass validation."""
    print("Test 59: Tenure=0 (schema min) should pass validation.")
    payload = dict(valid_payload); payload["Tenure"] = 0
    start = time.perf_counter()
    errors = validate_submission(payload, sample_schema)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert not any("Tenure" in e for e in errors)
    print(f"  RESULT: Tenure=0 (new customer) accepted by intake form in {elapsed_ms:.3f} ms")


def test_validate_tenure_max_is_valid_upper_bound(sample_schema, valid_payload):
    """Tenure == 61 (schema max) must pass validation."""
    print("Test 60: Tenure=61 (schema max) should pass validation.")
    payload = dict(valid_payload); payload["Tenure"] = 61
    start = time.perf_counter()
    errors = validate_submission(payload, sample_schema)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert not any("Tenure" in e for e in errors)
    print(f"  RESULT: Tenure=61 (max tenure) accepted by intake form in {elapsed_ms:.3f} ms")


def test_validate_float_for_int_field_triggers_error(sample_schema, valid_payload):
    """Passing 2.5 for an is_int field must produce an integer error."""
    print("Test 61: Non-integer value for is_int field should produce an error.")
    payload = dict(valid_payload); payload["CityTier"] = 2.5
    start = time.perf_counter()
    errors = validate_submission(payload, sample_schema)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert any("CityTier" in e for e in errors)
    print(f"  RESULT: CityTier=2.5 rejected as non-integer in {elapsed_ms:.3f} ms — form enforces integer-only fields")


def test_validate_none_churn_accepted_for_new_customer(sample_schema, valid_payload):
    """Churn=None must be accepted without error for new customers."""
    print("Test 62: Churn=None should be accepted for new customers.")
    payload = dict(valid_payload); payload["Churn"] = None
    start = time.perf_counter()
    errors = validate_submission(payload, sample_schema)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert not any("Churn" in e for e in errors)
    print(f"  RESULT: Churn=None accepted for new customer intake in {elapsed_ms:.3f} ms — label is optional on submission form")


def test_validate_customer_id_zero_is_invalid(sample_schema, valid_payload):
    """CustomerID <= 0 must be flagged as invalid."""
    print("Test 63: CustomerID=0 should be flagged as invalid.")
    payload = dict(valid_payload); payload["CustomerID"] = 0
    start = time.perf_counter()
    errors = validate_submission(payload, sample_schema)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert any("CustomerID" in e for e in errors)
    print(f"  RESULT: CustomerID=0 rejected in {elapsed_ms:.3f} ms — auto-ID assignment prevents zero IDs entering Firestore")


def test_validate_customer_id_negative_is_invalid(sample_schema, valid_payload):
    """Negative CustomerID must be flagged."""
    print("Test 64: Negative CustomerID should be flagged as invalid.")
    payload = dict(valid_payload); payload["CustomerID"] = -5
    start = time.perf_counter()
    errors = validate_submission(payload, sample_schema)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert any("CustomerID" in e for e in errors)
    print(f"  RESULT: CustomerID=-5 rejected in {elapsed_ms:.3f} ms — negative IDs blocked from customer database")


def test_validate_customer_id_numeric_string_passes(sample_schema, valid_payload):
    """CustomerID as a numeric string must pass (coercible to int)."""
    print("Test 65: CustomerID as numeric string should be accepted.")
    payload = dict(valid_payload); payload["CustomerID"] = "1006"
    start = time.perf_counter()
    errors = validate_submission(payload, sample_schema)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert not any("CustomerID" in e for e in errors)
    print(f"  RESULT: CustomerID='1006' (string) coerced and accepted in {elapsed_ms:.3f} ms")


def test_validate_whitespace_only_categorical_rejected(sample_schema, valid_payload):
    """A whitespace-only string for a categorical field must be rejected."""
    print("Test 66: Whitespace-only categorical value should be rejected.")
    payload = dict(valid_payload); payload["PreferredLoginDevice"] = "   "
    start = time.perf_counter()
    errors = validate_submission(payload, sample_schema)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert any("PreferredLoginDevice" in e for e in errors)
    print(f"  RESULT: Whitespace-only dropdown value rejected in {elapsed_ms:.3f} ms — form treats blank spaces as missing input")


def test_validate_extra_payload_keys_ignored(sample_schema, valid_payload):
    """Extra keys not in the schema must not cause validation errors."""
    print("Test 67: Extra payload keys outside schema should be ignored.")
    payload = dict(valid_payload); payload["UnknownField"] = "surprise"
    start = time.perf_counter()
    errors = validate_submission(payload, sample_schema)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert errors == []
    print(f"  RESULT: Extra field 'UnknownField' silently ignored in {elapsed_ms:.3f} ms — validation is schema-scoped only")


def test_validate_complain_zero_passes(sample_schema, valid_payload):
    """Complain=0 is a valid lower-bound integer value."""
    print("Test 68: Complain=0 should pass numeric validation.")
    payload = dict(valid_payload); payload["Complain"] = 0
    start = time.perf_counter()
    errors = validate_submission(payload, sample_schema)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert not any("Complain" in e for e in errors)
    print(f"  RESULT: Complain=0 (no complaints) accepted by intake form in {elapsed_ms:.3f} ms")


# --- get_next_customer_id — edge cases ---

def test_get_next_id_empty_collection_returns_one(fake_firestore):
    """Empty current_customers collection must return CustomerID = 1."""
    print("Test 69: Empty collection should return next CustomerID of 1.")
    from conftest import FakeDB
    db = FakeDB(); db.collection("current_customers")
    start = time.perf_counter()
    result = get_next_customer_id(db, fake_firestore)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert result == 1
    print(f"  RESULT: First ever customer auto-assigned CustomerID={result} in {elapsed_ms:.3f} ms")


def test_get_next_id_returns_highest_plus_one(fake_db, fake_firestore):
    """Seeded IDs 1001 and 1005 — next must be 1006."""
    print("Test 70: Seeded IDs 1001/1005 should produce next CustomerID of 1006.")
    start = time.perf_counter()
    result = get_next_customer_id(fake_db, fake_firestore)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert result == 1006
    print(f"  RESULT: Next CustomerID={result} (highest existing 1005 + 1) computed in {elapsed_ms:.3f} ms")


def test_get_next_id_single_record(fake_firestore):
    """Single record with CustomerID=42 must return 43."""
    print("Test 71: Single record CustomerID=42 should return 43.")
    from conftest import FakeDB
    db = FakeDB(); db.collection("current_customers").documents = {"doc_1": {"CustomerID": 42}}
    start = time.perf_counter()
    result = get_next_customer_id(db, fake_firestore)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert result == 43
    print(f"  RESULT: Next CustomerID={result} after single existing record (42) in {elapsed_ms:.3f} ms")


def test_get_next_id_non_integer_falls_back_gracefully(fake_firestore):
    """Non-integer CustomerID in Firestore must return 1 without crashing."""
    print("Test 72: Non-integer CustomerID should fall back to returning 1.")
    from conftest import FakeDB
    db = FakeDB(); db.collection("current_customers").documents = {"doc_1": {"CustomerID": "not-a-number"}}
    start = time.perf_counter()
    result = get_next_customer_id(db, fake_firestore)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert isinstance(result, int)
    print(f"  RESULT: Corrupt CustomerID in Firestore handled safely, fallback ID={result} in {elapsed_ms:.3f} ms")


# --- write_to_firestore — additional scenario coverage ---

def test_write_two_valid_submissions_accumulate(fake_db, fake_firestore, valid_payload):
    """Two valid submissions must add records to both collections."""
    print("Test 73: Two valid submissions should accumulate in both collections.")
    start = time.perf_counter()
    for _ in range(2):
        write_to_firestore(
            db=fake_db, firestore_module=fake_firestore,
            payload=dict(valid_payload), errors=[], source_value="batch_test",
        )
    elapsed_ms = (time.perf_counter() - start) * 1000
    subs = len(fake_db.collection("customer_submissions").documents)
    curr = len(fake_db.collection("current_customers").documents)
    assert subs == 2
    assert curr == 4
    print(f"  RESULT: 2 valid submissions written in {elapsed_ms:.2f} ms — customer_submissions={subs}, current_customers={curr}")


def test_write_invalid_then_valid_correct_counts(fake_db, fake_firestore, valid_payload):
    """Invalid then valid: submissions=2, current_customers gains only 1."""
    print("Test 74: Invalid then valid submission should add only 1 to current_customers.")
    start = time.perf_counter()
    write_to_firestore(db=fake_db, firestore_module=fake_firestore,
                       payload=dict(valid_payload), errors=["SomeField: required"], source_value="test")
    write_to_firestore(db=fake_db, firestore_module=fake_firestore,
                       payload=dict(valid_payload), errors=[], source_value="test")
    elapsed_ms = (time.perf_counter() - start) * 1000
    subs = len(fake_db.collection("customer_submissions").documents)
    curr = len(fake_db.collection("current_customers").documents)
    assert subs == 2
    assert curr == 3
    print(f"  RESULT: Invalid+valid pair processed in {elapsed_ms:.2f} ms — customer_submissions={subs} (all logged), current_customers={curr} (only valid promoted)")


def test_write_valid_stores_status_valid(fake_db, fake_firestore, valid_payload):
    """Valid submission document must have validation_status == 'valid'."""
    print("Test 75: Valid submission should store validation_status='valid'.")
    start = time.perf_counter()
    sid, _ = write_to_firestore(db=fake_db, firestore_module=fake_firestore,
                                 payload=dict(valid_payload), errors=[], source_value="test")
    elapsed_ms = (time.perf_counter() - start) * 1000
    doc = fake_db.collection("customer_submissions").documents[sid]
    assert doc["validation_status"] == "valid"
    print(f"  RESULT: Submission {sid} stored with status='{doc['validation_status']}' in {elapsed_ms:.3f} ms")


def test_write_invalid_stores_status_and_errors(fake_db, fake_firestore, valid_payload):
    """Invalid submission must store status='invalid' and the error list."""
    print("Test 76: Invalid submission should store status='invalid' and errors list.")
    errs = ["Tenure: out of accepted range"]
    start = time.perf_counter()
    sid, _ = write_to_firestore(db=fake_db, firestore_module=fake_firestore,
                                 payload=dict(valid_payload), errors=errs, source_value="test")
    elapsed_ms = (time.perf_counter() - start) * 1000
    doc = fake_db.collection("customer_submissions").documents[sid]
    assert doc["validation_status"] == "invalid"
    assert doc["validation_errors"] == errs
    print(f"  RESULT: Invalid submission {sid} stored with status='{doc['validation_status']}' and {len(errs)} error(s) logged in {elapsed_ms:.3f} ms")


def test_write_promoted_doc_back_references_submission_id(fake_db, fake_firestore, valid_payload):
    """Promoted record must contain the submission_id of its intake document."""
    print("Test 77: Promoted record should back-reference its submission_id.")
    start = time.perf_counter()
    sid, pid = write_to_firestore(db=fake_db, firestore_module=fake_firestore,
                                   payload=dict(valid_payload), errors=[], source_value="test")
    elapsed_ms = (time.perf_counter() - start) * 1000
    promoted = fake_db.collection("current_customers").documents[pid]
    assert promoted["submission_id"] == sid
    print(f"  RESULT: current_customers record {pid} correctly links back to submission {sid} in {elapsed_ms:.3f} ms")


def test_write_source_value_stored_on_promoted_record(fake_db, fake_firestore, valid_payload):
    """The source_value passed to write_to_firestore must appear on the promoted record."""
    print("Test 78: source_value should be stored on the promoted current_customers record.")
    start = time.perf_counter()
    _, pid = write_to_firestore(db=fake_db, firestore_module=fake_firestore,
                                 payload=dict(valid_payload), errors=[], source_value="streamlit_form")
    elapsed_ms = (time.perf_counter() - start) * 1000
    promoted = fake_db.collection("current_customers").documents[pid]
    assert promoted["source"] == "streamlit_form"
    print(f"  RESULT: Source tag 'streamlit_form' persisted on promoted record in {elapsed_ms:.3f} ms — data lineage traceable")


def test_write_invalid_promoted_id_is_none(fake_db, fake_firestore, valid_payload):
    """Invalid submission must return promoted_id=None."""
    print("Test 79: Invalid submission should return promoted_id=None.")
    start = time.perf_counter()
    _, pid = write_to_firestore(db=fake_db, firestore_module=fake_firestore,
                                 payload=dict(valid_payload), errors=["some error"], source_value="test")
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert pid is None
    print(f"  RESULT: Invalid submission returned promoted_id=None in {elapsed_ms:.3f} ms — Firestore current_customers unchanged")