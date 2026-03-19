"""
Client submission frontend for new customer intake.

This app builds a form from the original Kaggle CSV schema,
validates submissions, and writes to Firestore collections:
- customer_submissions (raw intake + validation status)
- current_customers (validated, operational records)
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

try:
    import firebase_admin
    from firebase_admin import credentials, firestore

    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False


st.set_page_config(
    page_title="Customer Intake Frontend",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap');
      html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
      }
      .hero {
        background: radial-gradient(circle at 20% 20%, #f6e7b2 0%, #fef6e4 35%, #e8f4f8 100%);
        border: 1px solid #e8dcc0;
        border-radius: 14px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.8rem;
      }
      .subtle {
        color: #4b5563;
      }
      .stButton button {
        background-color: #0f766e;
        color: #ffffff;
        border: none;
        border-radius: 10px;
        font-weight: 600;
      }
      .stButton button:hover {
        background-color: #115e59;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

CSV_PATH = Path(__file__).resolve().parent / "E Commerce Dataset.csv"
TARGET_COL = "Churn"
SOURCE_VALUE = "client_submission"

SUBMISSIONS_COLLECTION = "customer_submissions"
CURRENT_COLLECTION = "current_customers"


@st.cache_data
def load_dataset_schema(csv_path: Path) -> dict[str, Any]:
    df = pd.read_csv(csv_path)

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    numeric_stats: dict[str, dict[str, Any]] = {}
    for c in numeric_cols:
        col = pd.to_numeric(df[c], errors="coerce").dropna()
        if col.empty:
            continue
        is_int = bool((col % 1 == 0).all())
        numeric_stats[c] = {
            "min": float(col.min()),
            "max": float(col.max()),
            "median": float(col.median()),
            "is_int": is_int,
        }

    categorical_values: dict[str, list[str]] = {}
    for c in categorical_cols:
        options = sorted(df[c].dropna().astype(str).unique().tolist())
        categorical_values[c] = options

    return {
        "columns": df.columns.tolist(),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "numeric_stats": numeric_stats,
        "categorical_values": categorical_values,
    }


@st.cache_resource
def get_firestore_client():
    if not FIREBASE_AVAILABLE:
        raise RuntimeError("firebase-admin is not installed. Install it with: pip install firebase-admin")

    if not firebase_admin._apps:
        service_account = st.secrets.get("gcp_service_account", None)

        if service_account is None:
            raise RuntimeError("Missing gcp_service_account in Streamlit secrets.")

        service_account = dict(service_account)

        if "private_key" in service_account:
            service_account["private_key"] = service_account["private_key"].replace("\\n", "\n")

        cred = credentials.Certificate(service_account)
        firebase_admin.initialize_app(cred)

    return firestore.client()


def get_next_customer_id(db) -> int:
    docs = (
        db.collection(CURRENT_COLLECTION)
        .order_by("CustomerID", direction=firestore.Query.DESCENDING)
        .limit(1)
        .stream()
    )

    for doc in docs:
        data = doc.to_dict() or {}
        current_id = data.get("CustomerID", 0)
        try:
            return int(current_id) + 1
        except (TypeError, ValueError):
            return 1

    return 1


def validate_submission(payload: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    for col in schema["columns"]:
        value = payload.get(col)

        if col == TARGET_COL and value is None:
            continue

        if value is None or (isinstance(value, str) and not value.strip()):
            errors.append(f"{col}: value is required")
            continue

        if col == "CustomerID":
            try:
                customer_id = int(value)
                if customer_id <= 0:
                    errors.append("CustomerID: must be a positive integer")
            except (TypeError, ValueError):
                errors.append("CustomerID: must be an integer")
            continue

        if col in schema["numeric_cols"]:
            stats = schema["numeric_stats"].get(col)
            if stats is None:
                continue

            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                errors.append(f"{col}: must be numeric")
                continue

            if numeric_value < stats["min"] or numeric_value > stats["max"]:
                errors.append(
                    f"{col}: out of accepted range [{stats['min']:.2f}, {stats['max']:.2f}]"
                )

            if stats["is_int"] and float(numeric_value).is_integer() is False:
                errors.append(f"{col}: must be an integer")

        elif col in schema["categorical_cols"]:
            allowed = schema["categorical_values"].get(col, [])
            if str(value) not in allowed:
                errors.append(f"{col}: invalid category '{value}'")

    return errors


def write_to_firestore(db, payload: dict[str, Any], errors: list[str]) -> tuple[str, str | None]:
    now = firestore.SERVER_TIMESTAMP

    submission_doc = {
        **payload,
        "source": SOURCE_VALUE,
        "submitted_at": now,
        "validation_status": "valid" if not errors else "invalid",
        "validation_errors": errors,
    }

    created = db.collection(SUBMISSIONS_COLLECTION).add(submission_doc)
    submission_id = created[1].id

    promoted_id: str | None = None
    if not errors:
        promoted_doc = {
            **payload,
            "source": SOURCE_VALUE,
            "promoted_at": now,
            "submission_id": submission_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        promoted_ref = db.collection(CURRENT_COLLECTION).document(submission_id)
        promoted_ref.set(promoted_doc)
        promoted_id = submission_id

    return submission_id, promoted_id


def build_form(schema: dict[str, Any], next_customer_id: int) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    columns = schema["columns"]

    left, right = st.columns(2)
    visible_idx = 0

    for col in columns:
        if col == "CustomerID":
            payload[col] = next_customer_id
            continue

        container = left if visible_idx % 2 == 0 else right
        visible_idx += 1

        with container:
            if col == TARGET_COL:
                selected = st.selectbox(
                    f"{col} (optional for new customers)",
                    options=["Unknown", 0, 1],
                    index=0,
                    help="For fresh submissions, choose Unknown. Set 0/1 only if a known label exists.",
                )
                payload[col] = None if selected == "Unknown" else int(selected)
                continue

            if col in schema["numeric_cols"]:
                stats = schema["numeric_stats"][col]
                if stats["is_int"]:
                    value = st.number_input(
                        col,
                        min_value=int(stats["min"]),
                        max_value=int(stats["max"]),
                        value=int(stats["median"]),
                        step=1,
                    )
                    payload[col] = int(value)
                else:
                    value = st.number_input(
                        col,
                        min_value=float(stats["min"]),
                        max_value=float(stats["max"]),
                        value=float(stats["median"]),
                        step=0.01,
                    )
                    payload[col] = float(value)
            else:
                options = schema["categorical_values"].get(col, [])
                if options:
                    payload[col] = st.selectbox(col, options=options)
                else:
                    payload[col] = st.text_input(col, value="")

    return payload


def main():
    st.markdown(
        """
        <div class="hero">
          <h2 style="margin:0;">Live Customer Intake Form</h2>
          <p class="subtle" style="margin:0.35rem 0 0 0;">
            Captures the original Kaggle schema, validates inputs, then writes to Firestore.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.header("Pipeline Mapping")
    st.sidebar.caption("Phase 2: client form -> customer_submissions -> current_customers")
    st.sidebar.markdown(
        """
- Source tag: `client_submission`
- Invalid records stay in `customer_submissions`
- Valid records are promoted to `current_customers`
- CustomerID is auto-generated from Firestore
- Credentials should come from Streamlit secrets
        """
    )

    if not CSV_PATH.exists():
        st.error(f"CSV file not found: {CSV_PATH}")
        st.stop()

    schema = load_dataset_schema(CSV_PATH)

    try:
        db = get_firestore_client()
        next_customer_id = get_next_customer_id(db)
    except Exception as ex:
        st.error("Could not connect to Firestore to generate the next CustomerID.")
        st.code(str(ex))
        st.stop()

    with st.form("new_customer_form"):
        payload = build_form(schema, next_customer_id)
        submitted = st.form_submit_button("Submit New Customer")

    if not submitted:
        return

    validation_errors = validate_submission(payload, schema)

    if validation_errors:
        st.error("Validation failed. Record is captured in submissions but will not be promoted.")
        for e in validation_errors:
            st.write(f"- {e}")
    else:
        st.success("Validation passed.")

    try:
        submission_id, promoted_id = write_to_firestore(db, payload, validation_errors)

        st.info(f"Assigned CustomerID: {payload['CustomerID']}")
        st.info(f"Written to '{SUBMISSIONS_COLLECTION}' with document ID: {submission_id}")
        if promoted_id:
            st.success(f"Promoted to '{CURRENT_COLLECTION}' with document ID: {promoted_id}")

    except Exception as ex:
        st.warning("Could not write to Firestore yet. Check dependencies and secrets configuration.")
        st.code(str(ex))


if __name__ == "__main__":
    main()