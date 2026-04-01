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

# Human-readable labels for each column
FIELD_LABELS: dict[str, str] = {
    "Tenure":                     "Tenure (months with platform)",
    "PreferredLoginDevice":       "Preferred Login Device",
    "CityTier":                   "City Tier (1 = Metro, 3 = Small city)",
    "WarehouseToHome":            "Warehouse to Home Distance (km)",
    "PreferredPaymentMode":       "Preferred Payment Method",
    "Gender":                     "Gender",
    "HourSpendOnApp":             "Hours Spent on App (per day)",
    "NumberOfDeviceRegistered":   "Number of Devices Registered",
    "PreferedOrderCat":           "Preferred Order Category",
    "SatisfactionScore":          "Satisfaction Score (1-5)",
    "MaritalStatus":              "Marital Status",
    "NumberOfAddress":            "Number of Saved Addresses",
    "Complain":                   "Raised a Complaint? (0 = No, 1 = Yes)",
    "OrderAmountHikeFromlastYear":"Order Amount Increase from Last Year (%)",
    "CouponUsed":                 "Coupons Used (last month)",
    "OrderCount":                 "Number of Orders (last month)",
    "DaySinceLastOrder":          "Days Since Last Order",
    "CashbackAmount":             "Cashback Received (last month, $)",
    "Churn":                      "Churn Status",
}

FIELD_HELP: dict[str, str] = {
    "Tenure":                     "How many months the customer has been on the platform.",
    "PreferredLoginDevice":       "The device the customer most often uses to log in.",
    "CityTier":                   "Tier 1 = major metro, Tier 2 = mid-sized city, Tier 3 = small city or town.",
    "WarehouseToHome":            "Distance in km from the nearest fulfilment warehouse to the customer's address.",
    "PreferredPaymentMode":       "The payment method the customer uses most frequently.",
    "Gender":                     "Customer's gender.",
    "HourSpendOnApp":             "Average number of hours the customer spends on the mobile app or website per day.",
    "NumberOfDeviceRegistered":   "Total number of devices (phone, tablet, laptop) the customer has registered.",
    "PreferedOrderCat":           "The product category the customer orders most often.",
    "SatisfactionScore":          "Customer satisfaction rating from 1 (very dissatisfied) to 5 (very satisfied).",
    "MaritalStatus":              "Customer's marital status.",
    "NumberOfAddress":            "Number of delivery addresses saved in the customer's account.",
    "Complain":                   "Whether the customer raised a complaint in the last month (0 = No, 1 = Yes).",
    "OrderAmountHikeFromlastYear":"Percentage increase in the customer's order value compared to the same period last year.",
    "CouponUsed":                 "Number of discount coupons the customer used in the last month.",
    "OrderCount":                 "Total number of orders placed by the customer in the last month.",
    "DaySinceLastOrder":          "Number of days since the customer last placed an order.",
    "CashbackAmount":             "Total cashback amount (in $) the customer received in the last month.",
    "Churn":                      "Whether this customer has already churned. Leave as Unknown for new customers.",
}

# Group columns into sections for a cleaner layout
SECTION_GROUPS: list[tuple[str, list[str]]] = [
    ("Account Information", [
        "PreferredLoginDevice",
        "Tenure",
        "CityTier",
        "MaritalStatus",
        "Gender",
    ]),
    ("Shopping Behaviour", [
        "PreferedOrderCat",
        "PreferredPaymentMode",
        "HourSpendOnApp",
        "NumberOfDeviceRegistered",
        "NumberOfAddress",
    ]),
    ("Order Activity", [
        "OrderCount",
        "OrderAmountHikeFromlastYear",
        "CouponUsed",
        "DaySinceLastOrder",
        "CashbackAmount",
        "WarehouseToHome",
    ]),
    ("Satisfaction & Complaints", [
        "SatisfactionScore",
        "Complain",
    ]),
    ("Churn Label", [
        "Churn",
    ]),
]


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
            label = FIELD_LABELS.get(col, col)
            errors.append(f"{label}: value is required")
            continue

        if col == "CustomerID":
            try:
                customer_id = int(value)
                if customer_id <= 0:
                    errors.append("Customer ID: must be a positive integer")
            except (TypeError, ValueError):
                errors.append("Customer ID: must be an integer")
            continue

        if col in schema["numeric_cols"]:
            stats = schema["numeric_stats"].get(col)
            if stats is None:
                continue

            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                label = FIELD_LABELS.get(col, col)
                errors.append(f"{label}: must be numeric")
                continue

            if numeric_value < stats["min"] or numeric_value > stats["max"]:
                label = FIELD_LABELS.get(col, col)
                errors.append(
                    f"{label}: out of accepted range [{stats['min']:.2f}, {stats['max']:.2f}]"
                )

            if stats["is_int"] and float(numeric_value).is_integer() is False:
                label = FIELD_LABELS.get(col, col)
                errors.append(f"{label}: must be a whole number")

        elif col in schema["categorical_cols"]:
            allowed = schema["categorical_values"].get(col, [])
            if str(value) not in allowed:
                label = FIELD_LABELS.get(col, col)
                errors.append(f"{label}: invalid selection '{value}'")

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


def render_field(col: str, schema: dict[str, Any], payload: dict[str, Any]) -> None:
    label = FIELD_LABELS.get(col, col)
    help_text = FIELD_HELP.get(col, None)

    if col == TARGET_COL:
        selected = st.selectbox(
            label,
            options=["Unknown", "0 - Did not churn", "1 - Churned"],
            index=0,
            help="Leave as Unknown for new customers. Set only if the churn outcome is already known.",
        )
        if selected == "Unknown":
            payload[col] = None
        elif selected.startswith("0"):
            payload[col] = 0
        else:
            payload[col] = 1
        return

    if col in schema["numeric_cols"]:
        stats = schema["numeric_stats"][col]
        if stats["is_int"]:
            value = st.number_input(
                label,
                min_value=int(stats["min"]),
                max_value=int(stats["max"]),
                value=int(stats["median"]),
                step=1,
                help=help_text,
            )
            payload[col] = int(value)
        else:
            value = st.number_input(
                label,
                min_value=float(stats["min"]),
                max_value=float(stats["max"]),
                value=float(stats["median"]),
                step=0.01,
                help=help_text,
            )
            payload[col] = float(value)
    else:
        options = schema["categorical_values"].get(col, [])
        if options:
            payload[col] = st.selectbox(label, options=options, help=help_text)
        else:
            payload[col] = st.text_input(label, value="", help=help_text)


def build_form(schema: dict[str, Any], next_customer_id: int) -> dict[str, Any]:
    payload: dict[str, Any] = {"CustomerID": next_customer_id}

    st.info(f"Customer ID will be automatically assigned: **{next_customer_id}**")

    for section_title, cols in SECTION_GROUPS:
        st.markdown(f"#### {section_title}")
        left, right = st.columns(2)
        for i, col in enumerate(cols):
            if col not in schema["columns"]:
                continue
            with (left if i % 2 == 0 else right):
                render_field(col, schema, payload)

        st.divider()

    return payload


def main():
    st.markdown(
        """
        <div class="hero">
          <h2 style="margin:0;">New Customer Intake</h2>
          <p class="subtle" style="margin:0.35rem 0 0 0;">
            Fill in the customer details below. All fields are required unless marked optional.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.header("How it works")
    st.sidebar.markdown(
        """
- Fill in the form and click **Submit**
- Valid records are saved and available for churn scoring on the dashboard
- Invalid records are logged but not promoted
- Customer ID is assigned automatically
        """
    )

    if not CSV_PATH.exists():
        st.error(f"Reference dataset not found: {CSV_PATH}")
        st.stop()

    schema = load_dataset_schema(CSV_PATH)

    try:
        db = get_firestore_client()
        next_customer_id = get_next_customer_id(db)
    except Exception as ex:
        st.error("Could not connect to Firestore to generate the next Customer ID.")
        st.code(str(ex))
        st.stop()

    with st.form("new_customer_form"):
        payload = build_form(schema, next_customer_id)
        submitted = st.form_submit_button("Submit New Customer", use_container_width=True)

    if not submitted:
        return

    validation_errors = validate_submission(payload, schema)

    if validation_errors:
        st.error("Some fields have errors. The record has been logged but will not be scored on the dashboard.")
        for e in validation_errors:
            st.write(f"- {e}")
    else:
        st.success("Customer submitted successfully. They will appear on the dashboard after the next data refresh.")

    try:
        submission_id, promoted_id = write_to_firestore(db, payload, validation_errors)

        st.caption(f"Assigned Customer ID: {payload['CustomerID']}  |  Record ID: {submission_id}")

    except Exception as ex:
        st.warning("Could not write to Firestore. Check your secrets configuration.")
        st.code(str(ex))


if __name__ == "__main__":
    main()
