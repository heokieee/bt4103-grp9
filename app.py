"""
E-Commerce Churn Prediction Dashboard — Ensemble Deployment Version

Data sources:
1. Firestore live data from current_customers
2. Uploaded CSV

Model logic:
- Uses preprocessing pipeline exported from ensemble.ipynb
- Scores with Random Forest, XGBoost, and LightGBM
- Combines probabilities using weighted ensemble
"""

from __future__ import annotations

import time
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Optional
from google.api_core.exceptions import ResourceExhausted

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from drift_monitor import run_drift_monitor
from frontend_core import (
    CURRENT_COLLECTION as INTAKE_CURRENT_COLLECTION,
    SOURCE_VALUE as INTAKE_SOURCE_VALUE,
    SUBMISSIONS_COLLECTION as INTAKE_SUBMISSIONS_COLLECTION,
    TARGET_COL as INTAKE_TARGET_COL,
    get_next_customer_id as intake_get_next_customer_id,
    load_dataset_schema as intake_load_dataset_schema,
    validate_submission as intake_validate_submission,
    write_to_firestore as intake_write_to_firestore,
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

try:
    import firebase_admin
    from firebase_admin import credentials, firestore

    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False

try:
    from google.cloud import storage

    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="E-Commerce Churn Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #667eea11, #764ba211);
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    div[data-testid="stMetric"] label {
        font-size: 0.82rem !important;
        color: #555 !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 700 !important;
    }
    button[data-baseweb="tab"] {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        padding: 10px 24px !important;
    }
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }
    section[data-testid="stSidebar"] > div {
        padding-top: 1.5rem;
    }
    hr {
        border-color: #e8e8e8 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# PATHS / CONSTANTS
# =========================
BASE = Path(__file__).resolve().parent

PIPELINE_PATH = BASE / "ensemble_preprocessing_pipeline.joblib"
RF_MODEL_PATH = BASE / "ensemble_rf_model.joblib"
XGB_MODEL_PATH = BASE / "ensemble_xgb_model.joblib"
LGBM_MODEL_PATH = BASE / "ensemble_lgbm_model.joblib"
META_PATH = BASE / "ensemble_metadata.json"

METRICS_ALL_MODELS_PATH = BASE / "metrics_all_models.csv"
CONFUSION_ALL_MODELS_PATH = BASE / "confusion_all_models.csv"
FEATURE_IMPORTANCE_PATH = BASE / "ensemble_shap_feature_importance.csv"
REFERENCE_DATASET_PATH = BASE / "E Commerce Dataset.csv"
DRIFT_REPORT_PATH = BASE / "drift_report.csv"

CURRENT_COLLECTION = "current_customers"

DEFAULT_ID_CANDIDATES = [
    "CustomerID",
    "customerID",
    "customer_id",
    "id",
    "ID",
    "firestore_doc_id",
]

PLOTLY_TEMPLATE = "plotly_white"

ARTIFACT_PATHS = [
    PIPELINE_PATH,
    RF_MODEL_PATH,
    XGB_MODEL_PATH,
    LGBM_MODEL_PATH,
    META_PATH,
]

RISK_COLOURS = {
    "Low": "#2ecc71",
    "Medium": "#f39c12",
    "High": "#e74c3c",
    "Critical": "#8e44ad",
}


# =========================
# HELPERS
# =========================
def safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    return pd.read_csv(path) if path.exists() else None


def get_model_bucket_config() -> tuple[str, str]:
    bucket_name = str(st.secrets.get("model_bucket_name", "")).strip()
    bucket_prefix = str(st.secrets.get("model_bucket_prefix", "ensemble")).strip()

    # Fallback for TOML layouts where bucket keys were accidentally nested
    # under [gcp_service_account] in Streamlit secrets.
    if not bucket_name:
        service_account = st.secrets.get("gcp_service_account", None)
        if service_account is not None:
            bucket_name = str(service_account.get("model_bucket_name", "")).strip()
            bucket_prefix = str(service_account.get("model_bucket_prefix", bucket_prefix)).strip()

    return bucket_name, bucket_prefix


def build_blob_name(prefix: str, file_name: str) -> str:
    clean_prefix = prefix.strip().strip("/")
    if not clean_prefix:
        return file_name
    return f"{clean_prefix}/{file_name}"


def get_service_account_from_secrets() -> Optional[dict]:
    service_account = st.secrets.get("gcp_service_account", None)
    if service_account is None:
        return None

    service_account = dict(service_account)
    if "private_key" in service_account:
        service_account["private_key"] = service_account["private_key"].replace("\\n", "\n")
    return service_account


def sync_artifacts_from_bucket() -> tuple[bool, str]:
    bucket_name, bucket_prefix = get_model_bucket_config()
    if not bucket_name:
        return True, "model_bucket_name not configured; using local artifacts."

    if not GCS_AVAILABLE:
        return False, "google-cloud-storage is not installed in this environment."

    service_account = get_service_account_from_secrets()
    if service_account is None:
        return False, "Missing gcp_service_account in Streamlit secrets."

    client = storage.Client.from_service_account_info(service_account)
    bucket = client.bucket(bucket_name)

    downloaded: list[str] = []
    missing_in_bucket: list[str] = []
    for local_path in ARTIFACT_PATHS:
        blob_name = build_blob_name(bucket_prefix, local_path.name)
        blob = bucket.blob(blob_name)
        if not blob.exists(client):
            if local_path.exists():
                missing_in_bucket.append(blob_name)
                continue
            return False, f"Missing artifact in bucket and local disk: gs://{bucket_name}/{blob_name}"
        blob.download_to_filename(str(local_path))
        downloaded.append(local_path.name)

    if downloaded and missing_in_bucket:
        return (
            True,
            "Downloaded available artifacts from bucket: "
            + ", ".join(downloaded)
            + " | Missing in bucket (using local fallback): "
            + ", ".join(missing_in_bucket),
        )

    if downloaded:
        return True, "Downloaded latest artifacts from bucket: " + ", ".join(downloaded)

    if missing_in_bucket:
        return True, "Bucket artifacts missing; using local artifacts for now."

    return True, "No artifacts needed from bucket; using local artifacts."


def to_binary_series(series: pd.Series) -> pd.Series:
    s = series.copy()
    if s.dtype == object:
        mapping = {
            "Yes": 1, "No": 0,
            "yes": 1, "no": 0,
            "1": 1, "0": 0,
            1: 1, 0: 0,
            True: 1, False: 0,
        }
        s = s.map(mapping).fillna(s)
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)


def pick_customer_id_column(df: pd.DataFrame) -> Optional[str]:
    for col in DEFAULT_ID_CANDIDATES:
        if col in df.columns:
            return col
    for col in df.columns:
        col_l = col.lower()
        if "customer" in col_l and "id" in col_l:
            return col
    return None


def risk_tier_from_proba(proba: np.ndarray) -> pd.Series:
    return pd.cut(
        proba,
        bins=[0, 0.25, 0.50, 0.75, 1.01],
        labels=["Low", "Medium", "High", "Critical"],
        include_lowest=True,
    )


def simple_action_recommendations(row: pd.Series) -> list[str]:
    actions = []

    if row.get("Complain", 0) == 1:
        actions.append("Follow up on recent complaint and resolve within 48 hours.")
    if row.get("SatisfactionScore", 5) <= 2:
        actions.append("Launch satisfaction recovery outreach with a personal touchpoint.")
    if row.get("CashbackAmount", 999) < 130:
        actions.append("Offer a cashback or retention incentive to improve stickiness.")
    if row.get("CouponUsed", 999) < 1:
        actions.append("Send a coupon campaign to encourage the next purchase.")
    if row.get("DaySinceLastOrder", 0) > 10:
        actions.append("Trigger a win-back campaign because the customer has been inactive.")
    if row.get("Tenure", 99) <= 2:
        actions.append("Prioritise onboarding and early lifecycle engagement.")

    if not actions:
        actions.append("No major immediate churn signals detected. Continue standard engagement.")

    return actions


# =========================
# FIRESTORE
# =========================
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


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_firestore_customers(collection_name: str = CURRENT_COLLECTION) -> pd.DataFrame:
    db = get_firestore_client()
    retry_delays = [1, 2, 4, 8]

    for attempt, delay in enumerate(retry_delays, start=1):
        try:
            print(f"FIRESTORE FETCH at {time.strftime('%H:%M:%S')}")
            docs = db.collection(collection_name).limit(50).stream()
            rows = []
            for doc in docs:
                row = doc.to_dict()
                row["firestore_doc_id"] = doc.id
                rows.append(row)

            if not rows:
                return pd.DataFrame()

            df = pd.DataFrame(rows)

            for col in ["submitted_at", "promoted_at", "created_at_utc"]:
                if col in df.columns:
                    df[col] = df[col].astype(str)

            if "CustomerID" not in df.columns and "firestore_doc_id" in df.columns:
                df["CustomerID"] = df["firestore_doc_id"]

            return df

        except ResourceExhausted:
            if attempt == len(retry_delays):
                raise
            time.sleep(delay)

    return pd.DataFrame()


# =========================
# ARTIFACT LOADING
# =========================
@st.cache_resource
def load_artifacts():
    pipeline = joblib.load(PIPELINE_PATH)
    rf_model = joblib.load(RF_MODEL_PATH)
    xgb_model = joblib.load(XGB_MODEL_PATH)
    lgbm_model = joblib.load(LGBM_MODEL_PATH)

    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return pipeline, rf_model, xgb_model, lgbm_model, meta


def preprocess_for_ensemble(
    df_input: pd.DataFrame,
    pipeline,
    selected_features: list[str],
) -> pd.DataFrame:
    X_processed = pipeline.transform(df_input)

    if isinstance(X_processed, pd.DataFrame):
        return X_processed

    X_processed = pd.DataFrame(X_processed)

    if len(selected_features) == X_processed.shape[1]:
        X_processed.columns = selected_features

    return X_processed


def predict_weighted_ensemble(
    X_processed: pd.DataFrame,
    rf_model,
    xgb_model,
    lgbm_model,
    w_rf: float,
    w_xgb: float,
    w_lgbm: float,
) -> np.ndarray:
    rf_proba = rf_model.predict_proba(X_processed)[:, 1]
    xgb_proba = xgb_model.predict_proba(X_processed)[:, 1]
    lgbm_proba = lgbm_model.predict_proba(X_processed)[:, 1]

    y_proba = (
        w_rf * rf_proba +
        w_xgb * xgb_proba +
        w_lgbm * lgbm_proba
    )
    return y_proba


def run_retraining_job() -> tuple[bool, str]:
    service_account = get_service_account_from_secrets()
    if service_account is None:
        return False, "Missing gcp_service_account in Streamlit secrets."

    retrain_script = BASE / "retrain.py"
    if not retrain_script.exists():
        return False, f"Retraining script not found: {retrain_script}"

    bucket_name, bucket_prefix = get_model_bucket_config()
    print(
        "Retrain bucket config:",
        f"bucket_name={'<empty>' if not bucket_name else bucket_name}",
        f"bucket_prefix={bucket_prefix}",
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=True) as temp_sa:
        json.dump(service_account, temp_sa)
        temp_sa.flush()

        cmd = [sys.executable, str(retrain_script), "--service-account", temp_sa.name]
        if bucket_name:
            cmd.extend(["--bucket", bucket_name, "--bucket-prefix", bucket_prefix])
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(BASE))

    output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")

    # Mirror subprocess output to server logs for Streamlit Cloud debugging.
    print("\n" + "=" * 30)
    print("RETRAIN SUBPROCESS OUTPUT")
    print("=" * 30)
    print(output.strip() or "<empty>")
    print("=" * 30 + "\n")

    if proc.returncode != 0:
        return False, output.strip() or "Retraining failed without logs."

    load_artifacts.clear()
    fetch_firestore_customers.clear()
    return True, output.strip() or "Retraining completed."


def build_intake_form(schema: dict[str, object], next_customer_id: int) -> dict[str, object]:
    payload: dict[str, object] = {}
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
            if col == INTAKE_TARGET_COL:
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


def render_intake_mode() -> None:
    st.markdown(
        """
        # New Customer Intake
        Submit a customer profile and write it to Firestore.
        """
    )

    st.caption(
        "Writes to customer_submissions, then promotes valid records to current_customers."
    )

    if not REFERENCE_DATASET_PATH.exists():
        st.error(f"CSV file not found: {REFERENCE_DATASET_PATH}")
        return

    schema = intake_load_dataset_schema(REFERENCE_DATASET_PATH)

    try:
        db = get_firestore_client()
        next_customer_id = intake_get_next_customer_id(
            db,
            firestore_module=firestore,
            current_collection=INTAKE_CURRENT_COLLECTION,
        )
    except Exception as ex:
        st.error("Could not connect to Firestore to generate the next CustomerID.")
        st.code(str(ex))
        return

    with st.form("new_customer_form"):
        payload = build_intake_form(schema, next_customer_id)
        submitted = st.form_submit_button("Submit New Customer")

    if not submitted:
        return

    validation_errors = intake_validate_submission(
        payload,
        schema,
        target_col=INTAKE_TARGET_COL,
    )

    if validation_errors:
        st.error("Validation failed. Record is captured in submissions but will not be promoted.")
        for err in validation_errors:
            st.write(f"- {err}")
    else:
        st.success("Validation passed.")

    try:
        submission_id, promoted_id = intake_write_to_firestore(
            db,
            firestore_module=firestore,
            payload=payload,
            errors=validation_errors,
            source_value=INTAKE_SOURCE_VALUE,
            submissions_collection=INTAKE_SUBMISSIONS_COLLECTION,
            current_collection=INTAKE_CURRENT_COLLECTION,
        )

        st.info(f"Assigned CustomerID: {payload['CustomerID']}")
        st.info(
            "Written to '{}' with document ID: {}".format(
                INTAKE_SUBMISSIONS_COLLECTION,
                submission_id,
            )
        )
        if promoted_id:
            st.success(
                "Promoted to '{}' with document ID: {}".format(
                    INTAKE_CURRENT_COLLECTION,
                    promoted_id,
                )
            )

        if st.button("Go to Dashboard", use_container_width=True):
            st.session_state["app_mode"] = "Dashboard"
            st.rerun()

    except Exception as ex:
        st.warning("Could not write to Firestore yet. Check dependencies and secrets configuration.")
        st.code(str(ex))


# =========================
# HEADER
# =========================
st.markdown("# E-Commerce Customer Intelligence App")

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("## App Mode")
    app_mode = st.radio(
        "Select Experience",
        options=["Dashboard", "New Customer Intake"],
        index=0,
        key="app_mode",
    )

    st.markdown("---")

    data_source = "Firestore Live Data"
    threshold = 0.50
    uploaded = None

    if app_mode == "Dashboard":
        st.markdown("## Dashboard Controls")
        st.markdown("---")

        data_source = st.radio(
            "Data Source",
            options=["Firestore Live Data", "Upload CSV"],
            index=0,
        )

        threshold = st.slider(
            "Churn Probability Threshold",
            min_value=0.05,
            max_value=0.95,
            value=0.50,
            step=0.01,
            help="Customers above this probability are classified as likely churners.",
        )

        st.markdown("---")
        if data_source == "Upload CSV":
            uploaded = st.file_uploader(
                "Upload Customer CSV",
                type=["csv"],
                help="Upload a CSV with the same raw columns used in ensemble.ipynb.",
            )

        st.markdown("---")
        with st.expander("Artifact Status", expanded=False):
            artifact_checks = {
                "Preprocessing Pipeline": PIPELINE_PATH.exists(),
                "Random Forest Model": RF_MODEL_PATH.exists(),
                "XGBoost Model": XGB_MODEL_PATH.exists(),
                "LightGBM Model": LGBM_MODEL_PATH.exists(),
                "Metadata": META_PATH.exists(),
                "Feature Importance": FEATURE_IMPORTANCE_PATH.exists(),
                "Firebase Admin SDK": FIREBASE_AVAILABLE,
                "GCS SDK": GCS_AVAILABLE,
            }
            for name, ok in artifact_checks.items():
                st.markdown("**{}:** {}".format(name, "Available" if ok else "Missing"))

        st.markdown("---")
        with st.expander("Model Maintenance", expanded=False):
            st.caption("Runs retraining with Firestore data and updates ensemble artifacts.")

            if st.button("Retrain Model", use_container_width=True):
                with st.spinner("Retraining model... this may take a minute."):
                    ok, logs = run_retraining_job()

                if ok:
                    st.success("Retraining completed and artifacts were refreshed.")
                    st.text_area("Retraining Output", value=logs, height=180)
                else:
                    st.error("Retraining failed.")
                    st.text_area("Retraining Output", value=logs, height=220)
    else:
        st.caption("Use this mode to capture and validate new customer records.")

if app_mode == "New Customer Intake":
    render_intake_mode()
    st.stop()

# =========================
# LOAD MODEL ARTIFACTS
# =========================
sync_ok, sync_message = sync_artifacts_from_bucket()
if not sync_ok:
    st.error("Could not sync latest model artifacts from bucket.")
    st.code(sync_message)
    st.stop()

try:
    pipeline, rf_model, xgb_model, lgbm_model, meta = load_artifacts()
except Exception as e:
    st.error("Failed to load ensemble artifacts. Make sure all exported notebook files are in the same folder as app.py.")
    st.exception(e)
    st.stop()

target_col = meta.get("target", "Churn")
raw_input_columns = meta.get("raw_input_columns", [])
selected_features = meta.get("selected_features", [])
weights = meta.get("weights", {})

w_rf = float(weights.get("w_rf", 1 / 3))
w_xgb = float(weights.get("w_xgb", 1 / 3))
w_lgbm = float(weights.get("w_lgbm", 1 / 3))

metrics_all = safe_read_csv(METRICS_ALL_MODELS_PATH)
conf_all = safe_read_csv(CONFUSION_ALL_MODELS_PATH)
feat_imp_external = safe_read_csv(FEATURE_IMPORTANCE_PATH)

# =========================
# LOAD DATA
# =========================
if data_source == "Firestore Live Data":
    try:
        db = get_firestore_client()
        st.write("Connected project:", db.project)

        df = fetch_firestore_customers(CURRENT_COLLECTION)
    except ResourceExhausted:
        st.error("Firestore quota exceeded (429). Please wait a few minutes and try again.")
        st.stop()
    except Exception as e:
        st.error("Could not fetch Firestore data.")
        st.exception(e)
        st.stop()

    if df.empty:
        st.warning("No records found in Firestore collection 'current_customers'.")
        st.stop()

else:
    if uploaded is None:
        st.markdown("---")
        st.info("Upload a CSV from the sidebar or switch to Firestore Live Data.")
        st.stop()

    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error("Could not read the uploaded CSV.")
        st.exception(e)
        st.stop()

# =========================
# PREPARE DATA
# =========================
has_target = target_col in df.columns
y_true = to_binary_series(df[target_col].copy()) if has_target else None

df_original = df.copy()
df_features = df.drop(columns=[target_col]).copy() if has_target else df.copy()

metadata_cols = [
    "source",
    "submission_id",
    "submitted_at",
    "promoted_at",
    "created_at_utc",
    "validation_status",
    "validation_errors",
    "firestore_doc_id",
]

for col in metadata_cols:
    if col in df_features.columns and col not in raw_input_columns:
        df_features = df_features.drop(columns=[col])

id_col_detected = pick_customer_id_column(df_features)

missing_for_model = [c for c in raw_input_columns if c not in df_features.columns]
if raw_input_columns and missing_for_model:
    st.error("Missing required feature columns for the ensemble model.")
    st.write(missing_for_model)
    st.stop()

if raw_input_columns:
    X_raw = df_features[raw_input_columns].copy()
else:
    X_raw = df_features.copy()

# =========================
# SCORE
# =========================
try:
    X_model = preprocess_for_ensemble(X_raw, pipeline, selected_features)
    proba = predict_weighted_ensemble(
        X_model,
        rf_model,
        xgb_model,
        lgbm_model,
        w_rf,
        w_xgb,
        w_lgbm,
    )
    pred = (proba >= threshold).astype(int)
except Exception as e:
    st.error("Prediction failed.")
    st.exception(e)
    st.stop()

out = df_original.copy()
out["Churn_probability"] = proba
out["Churn_pred"] = pred
out["Risk_Tier"] = risk_tier_from_proba(proba).astype(str)

# =========================
# FILTERS
# =========================
st.markdown("---")
with st.expander("Filters — Narrow by demographics and behaviour", expanded=False):
    filter_cols = [
        "Gender",
        "MaritalStatus",
        "PreferredLoginDevice",
        "PreferedOrderCat",
        "PreferredPaymentMode",
        "CityTier",
    ]
    available_filters = [c for c in filter_cols if c in out.columns]

    if not available_filters:
        st.caption("No standard filter columns found in this dataset.")
        filtered = out.copy()
    else:
        cols = st.columns(min(3, len(available_filters)))
        selected = {}

        for i, c in enumerate(available_filters):
            with cols[i % len(cols)]:
                opts = sorted([x for x in out[c].dropna().unique().tolist()])
                sel = st.multiselect(c, options=opts, default=opts)
                selected[c] = set(sel)

        mask = np.ones(len(out), dtype=bool)
        for c, keep in selected.items():
            mask &= out[c].isin(list(keep))

        filtered = out.loc[mask].copy()

    st.info("Showing {:,} of {:,} customers after filters.".format(len(filtered), len(out)))

# =========================
# KPI CARDS
# =========================
st.markdown("### Key Metrics")
k1, k2, k3, k4 = st.columns(4)

if has_target and target_col in filtered.columns:
    y_f = to_binary_series(filtered[target_col])
    churn_rate = float((y_f == 1).mean())
    k1.metric("Churn Rate (Actual)", "{:.1f}%".format(churn_rate * 100), "{} / {}".format(int((y_f == 1).sum()), len(y_f)))
else:
    churn_rate = float(filtered["Churn_pred"].mean())
    k1.metric("Churn Rate (Predicted)", "{:.1f}%".format(churn_rate * 100))

k2.metric("Customers at Risk", "{:,}".format(int(filtered["Churn_pred"].sum())))

if has_target and target_col in filtered.columns:
    y_f = to_binary_series(filtered[target_col])
    acc_f = accuracy_score(y_f, filtered["Churn_pred"])
    rec_f = recall_score(y_f, filtered["Churn_pred"], zero_division=0)
    k3.metric("Model Accuracy", "{:.3f}".format(acc_f))
    k4.metric("Recall", "{:.3f}".format(rec_f))
else:
    k3.metric("Weight RF", "{:.3f}".format(w_rf))
    k4.metric("Weight XGB / LGBM", "{:.3f} / {:.3f}".format(w_xgb, w_lgbm))

st.markdown("---")

drift_report_df = None
drift_error = None
try:
    drift_report_df = run_drift_monitor(
        reference_csv_path=REFERENCE_DATASET_PATH,
        current_df=df,
        output_path=DRIFT_REPORT_PATH,
    )
except Exception as ex:
    drift_error = str(ex)

# =========================
# TABS
# =========================
tab_model, tab_drivers, tab_segments, tab_lookup, tab_data, tab_drift = st.tabs([
    "Model Performance",
    "Top Churn Drivers",
    "Risk Segmentation",
    "Customer Lookup",
    "Scored Data",
    "Data Drift",
])

# =========================
# TAB 1: MODEL PERFORMANCE
# =========================
with tab_model:
    st.markdown("### Weighted Ensemble Performance")

    st.markdown(
        """
        This dashboard uses weighted ensemble probabilities from three chosen models:
        - Preprocessing pipeline with imputation, interaction features, scaling, one-hot encoding, and SelectKBest
        - Random Forest, XGBoost and LightGBM probability predictions
        - Weighted averaging using recall-derived weights
        """
    )

    weight_df = pd.DataFrame(
        {
            "Model": ["Random Forest", "XGBoost", "LightGBM"],
            "Weight": [w_rf, w_xgb, w_lgbm],
        }
    )
    st.dataframe(weight_df, use_container_width=True, hide_index=True)

    if has_target and target_col in filtered.columns:
        y_f = to_binary_series(filtered[target_col])
        roc = roc_auc_score(y_f, filtered["Churn_probability"]) if len(np.unique(y_f)) == 2 else None
        prec_v = precision_score(y_f, filtered["Churn_pred"], zero_division=0)
        rec_v = recall_score(y_f, filtered["Churn_pred"], zero_division=0)
        f1v = f1_score(y_f, filtered["Churn_pred"], zero_division=0)
        acc_v = accuracy_score(y_f, filtered["Churn_pred"])

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", "{:.4f}".format(acc_v))
        m2.metric("Precision", "{:.4f}".format(prec_v))
        m3.metric("Recall", "{:.4f}".format(rec_v))
        m4.metric("F1 Score", "{:.4f}".format(f1v))
        m5.metric("ROC AUC", "{:.4f}".format(roc) if roc is not None else "N/A")

        col_cm, col_roc = st.columns(2)

        with col_cm:
            cm = confusion_matrix(y_f, filtered["Churn_pred"])
            labels = ["Stayed", "Churned"]
            fig_cm = go.Figure(
                data=go.Heatmap(
                    z=cm[::-1],
                    x=labels,
                    y=labels[::-1],
                    text=cm[::-1],
                    texttemplate="%{text}",
                    textfont=dict(size=18),
                    colorscale="Blues",
                    showscale=False,
                )
            )
            fig_cm.update_layout(
                title="Confusion Matrix",
                xaxis_title="Predicted",
                yaxis_title="Actual",
                height=400,
                template=PLOTLY_TEMPLATE,
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        with col_roc:
            if roc is not None:
                fpr, tpr, _ = roc_curve(y_f, filtered["Churn_probability"])
                fig_roc = go.Figure()
                fig_roc.add_trace(
                    go.Scatter(
                        x=fpr,
                        y=tpr,
                        mode="lines",
                        name="Weighted Ensemble (AUC = {:.4f})".format(roc),
                        line=dict(width=3),
                    )
                )
                fig_roc.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        mode="lines",
                        name="Random",
                        line=dict(color="grey", dash="dash"),
                    )
                )
                fig_roc.update_layout(
                    title="ROC Curve",
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate",
                    height=400,
                    template=PLOTLY_TEMPLATE,
                    legend=dict(x=0.50, y=0.1),
                )
                st.plotly_chart(fig_roc, use_container_width=True)
            else:
                st.info("ROC curve requires both classes in the filtered data.")
    else:
        st.info("Live Firestore data is being scored for inference. Upload a CSV with a Churn column to view evaluation metrics.")

    if metrics_all is not None:
        st.markdown("---")
        st.markdown("### Model Comparison")
        metric_options = [c for c in ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"] if c in metrics_all.columns]

        if metric_options:
            fig_comp = go.Figure()
            colours = px.colors.qualitative.Set2

            for idx, metric_name in enumerate(metric_options):
                fig_comp.add_trace(
                    go.Bar(
                        x=metrics_all["Model"],
                        y=metrics_all[metric_name],
                        name=metric_name,
                        marker_color=colours[idx % len(colours)],
                    )
                )

            fig_comp.update_layout(
                barmode="group",
                title="Performance Metrics Across Models",
                yaxis_title="Score",
                height=420,
                template=PLOTLY_TEMPLATE,
                legend=dict(orientation="h", y=-0.15),
            )
            st.plotly_chart(fig_comp, use_container_width=True)

# =========================
# TAB 2: TOP CHURN DRIVERS
# =========================
with tab_drivers:
    st.markdown("### Top Churn Drivers")

    if feat_imp_external is not None and "feature" in feat_imp_external.columns:

        # -----------------------------------
        # Clean feature names
        # -----------------------------------
        def clean_feature_name(f):
            # Remove pipeline prefixes
            f = f.replace("num__", "").replace("cat__", "")

            # Handle one-hot encoded features
            if "_" in f:
                parts = f.split("_")
                # if it's categorical encoding (last part is category)
                if len(parts) > 1 and parts[-1].isalpha():
                    return f"{parts[0]} = {parts[-1]}"
            
            # Handle interaction features (space separated)
            if " " in f:
                return " × ".join(f.split(" "))
            
            return f

        feat_imp_external["clean_feature"] = feat_imp_external["feature"].apply(clean_feature_name)

        # -----------------------------------
        # Pick importance column
        # -----------------------------------
        if "ensemble_shap_importance" in feat_imp_external.columns:
            importance_col = "ensemble_shap_importance"
            chart_title = "Top Ensemble SHAP Drivers"
        elif "importance" in feat_imp_external.columns:
            importance_col = "importance"
            chart_title = "Top Features"
        else:
            importance_col = None

        if importance_col is not None:
            top_n = st.slider("Number of top features to display", 5, 40, 20, key="shap_top_n")

            fi = (
                feat_imp_external[["clean_feature", importance_col]]
                .dropna()
                .sort_values(importance_col, ascending=False)
                .head(top_n)
            )

            fig_shap = go.Figure(
                go.Bar(
                    x=fi[importance_col].values[::-1],
                    y=fi["clean_feature"].values[::-1],
                    orientation="h",
                )
            )

            fig_shap.update_layout(
                title=f"{chart_title} (Top {top_n})",
                xaxis_title="Mean |SHAP value|",
                yaxis_title="Feature",
                height=max(400, top_n * 24),
                template=PLOTLY_TEMPLATE,
                margin=dict(l=200),
            )

            st.plotly_chart(fig_shap, use_container_width=True)

            with st.expander("View feature importance table"):
                st.dataframe(fi.reset_index(drop=True), use_container_width=True)

        else:
            st.warning(
                "No valid importance column found. Expected 'ensemble_shap_importance' or 'importance'."
            )
    else:
        st.warning(
            "Feature importance CSV not found. Export 'ensemble_shap_feature_importance.csv' from notebook."
        )

# =========================
# TAB 3: RISK SEGMENTATION
# =========================
with tab_segments:
    st.markdown("### Customer Risk Segmentation")

    col_pie1, col_pie2 = st.columns(2)

    with col_pie1:
        tier_counts = filtered["Risk_Tier"].value_counts()
        fig_pie = go.Figure(
            go.Pie(
                labels=tier_counts.index.astype(str).tolist(),
                values=tier_counts.values.tolist(),
                hole=0.45,
                marker=dict(colors=[RISK_COLOURS.get(str(t), "#999") for t in tier_counts.index]),
                textinfo="label+percent",
                textfont=dict(size=14),
            )
        )
        fig_pie.update_layout(title="Risk Tier Distribution", height=420, template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_pie2:
        if "PreferedOrderCat" in filtered.columns:
            cat_counts = filtered["PreferedOrderCat"].astype(str).value_counts().head(10)
            fig_cat = go.Figure(
                go.Pie(
                    labels=cat_counts.index.tolist(),
                    values=cat_counts.values.tolist(),
                    hole=0.45,
                    textinfo="label+percent",
                    textfont=dict(size=14),
                )
            )
            fig_cat.update_layout(title="Top Order Categories", height=420, template=PLOTLY_TEMPLATE)
            st.plotly_chart(fig_cat, use_container_width=True)
        else:
            st.info("PreferedOrderCat not available.")

    st.markdown("---")
    st.markdown("#### Risk Tier Summary")
    tier_summary = filtered.groupby("Risk_Tier", observed=False).agg(
        Customers=("Churn_pred", "count"),
        Avg_Probability=("Churn_probability", "mean"),
        Predicted_Churners=("Churn_pred", "sum"),
    ).reset_index()
    tier_summary["Avg_Probability"] = tier_summary["Avg_Probability"].map("{:.2%}".format)
    tier_summary["Predicted_Churners"] = tier_summary["Predicted_Churners"].astype(int)
    st.dataframe(tier_summary, use_container_width=True, hide_index=True)

# =========================
# TAB 4: CUSTOMER LOOKUP
# =========================
with tab_lookup:
    st.markdown("### Individual Customer Lookup")

    id_col = pick_customer_id_column(filtered)
    if id_col is None:
        st.warning("No CustomerID column found. Add one to enable lookup.")
    else:
        st.caption("Using ID column: `{}`".format(id_col))

        ids = sorted(filtered[id_col].astype(str).unique().tolist())
        q = st.text_input("Search Customer ID", value="", placeholder="Type to filter...")
        ids_view = [i for i in ids if q.strip().lower() in i.lower()] if q.strip() else ids

        if not ids_view:
            st.info("No matching customer found.")
        else:
            chosen_id = st.selectbox("Select Customer", ids_view[:5000])
            row = filtered.loc[filtered[id_col].astype(str) == str(chosen_id)].iloc[0]

            prob = float(row["Churn_probability"])
            tier = str(row["Risk_Tier"])
            pred_label = int(row["Churn_pred"])

            st.markdown("---")
            h1, h2, h3 = st.columns(3)

            prob_colour = (
                "#2ecc71" if prob < 0.25
                else "#f39c12" if prob < 0.5
                else "#e74c3c" if prob < 0.75
                else "#8e44ad"
            )

            h1.markdown(
                """
                <div style="text-align:center; padding:20px; border-radius:12px; background:{0}15; border:2px solid {0};">
                    <div style="font-size:0.85rem; color:#666; font-weight:600;">CHURN PROBABILITY</div>
                    <div style="font-size:2.2rem; font-weight:700; color:{0};">{1:.1%}</div>
                </div>
                """.format(prob_colour, prob),
                unsafe_allow_html=True,
            )

            tier_colour = RISK_COLOURS.get(tier, "#999")
            h2.markdown(
                """
                <div style="text-align:center; padding:20px; border-radius:12px; background:{0}15; border:2px solid {0};">
                    <div style="font-size:0.85rem; color:#666; font-weight:600;">RISK TIER</div>
                    <div style="font-size:2.2rem; font-weight:700; color:{0};">{1}</div>
                </div>
                """.format(tier_colour, tier),
                unsafe_allow_html=True,
            )

            pred_colour = "#e74c3c" if pred_label == 1 else "#2ecc71"
            pred_text = "Will Churn" if pred_label == 1 else "Will Stay"
            h3.markdown(
                """
                <div style="text-align:center; padding:20px; border-radius:12px; background:{0}15; border:2px solid {0};">
                    <div style="font-size:0.85rem; color:#666; font-weight:600;">PREDICTION</div>
                    <div style="font-size:2.2rem; font-weight:700; color:{0};">{1}</div>
                </div>
                """.format(pred_colour, pred_text),
                unsafe_allow_html=True,
            )

            st.markdown("")

            col_snap, col_action = st.columns([3, 2])

            with col_snap:
                st.markdown("#### Customer Profile")
                preferred_cols = [
                    "Tenure",
                    "CityTier",
                    "WarehouseToHome",
                    "HourSpendOnApp",
                    "NumberOfDeviceRegistered",
                    "SatisfactionScore",
                    "NumberOfAddress",
                    "Complain",
                    "OrderAmountHikeFromlastYear",
                    "CouponUsed",
                    "OrderCount",
                    "DaySinceLastOrder",
                    "CashbackAmount",
                    "PreferredLoginDevice",
                    "PreferredPaymentMode",
                    "Gender",
                    "PreferedOrderCat",
                    "MaritalStatus",
                ]
                snapshot_cols = [c for c in preferred_cols if c in row.index]
                if snapshot_cols:
                    snapshot = pd.DataFrame(
                        {"Feature": snapshot_cols, "Value": [row[c] for c in snapshot_cols]}
                    )
                    st.dataframe(snapshot, use_container_width=True, hide_index=True)

            with col_action:
                st.markdown("#### Recommended Actions")
                for rec in simple_action_recommendations(row):
                    st.markdown("- {}".format(rec))

            st.markdown("---")
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    title={"text": "Churn Risk Score", "font": {"size": 16}},
                    number={"suffix": "%"},
                    gauge=dict(
                        axis=dict(range=[0, 100]),
                        bar=dict(color=prob_colour),
                        steps=[
                            dict(range=[0, 25], color="rgba(46, 204, 113, 0.2)"),
                            dict(range=[25, 50], color="rgba(243, 156, 18, 0.2)"),
                            dict(range=[50, 75], color="rgba(231, 76, 60, 0.2)"),
                            dict(range=[75, 100], color="rgba(142, 68, 173, 0.2)"),
                        ],
                        threshold=dict(
                            line=dict(color="black", width=3),
                            thickness=0.8,
                            value=threshold * 100,
                        ),
                    ),
                )
            )
            fig_gauge.update_layout(height=280, template=PLOTLY_TEMPLATE)
            st.plotly_chart(fig_gauge, use_container_width=True)

# =========================
# TAB 5: SCORED DATA
# =========================
with tab_data:
    st.markdown("### Scored Customer Data")

    s1, s2, s3 = st.columns(3)
    s1.metric("Total Rows", "{:,}".format(len(filtered)))
    s2.metric("Predicted Churners", "{:,}".format(int(filtered["Churn_pred"].sum())))
    s3.metric("Average Probability", "{:.2%}".format(filtered["Churn_probability"].mean()))

    st.markdown("")

    max_rows = max(10, min(500, len(filtered)))
    default_rows = min(100, len(filtered)) if len(filtered) > 0 else 10
    show_n = st.slider("Rows to preview", 10, max_rows, default_rows, key="data_rows")
    st.dataframe(filtered.head(show_n), use_container_width=True, hide_index=True)

    st.download_button(
        "Download Full Scored CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="scored_customers.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.markdown("#### Churn Probability Distribution")

    fig_hist = px.histogram(
        filtered,
        x="Churn_probability",
        nbins=40,
        labels={"Churn_probability": "Churn Probability"},
    )
    fig_hist.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="red",
        annotation_text="Threshold ({})".format(threshold),
    )
    fig_hist.update_layout(
        title="Distribution of Predicted Churn Probabilities",
        yaxis_title="Count",
        height=380,
        template=PLOTLY_TEMPLATE,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# =========================
# TAB 6: DATA DRIFT
# =========================
with tab_drift:
    st.markdown("### Data Drift")

    if drift_error is not None:
        st.error("Unable to compute drift report.")
        st.code(drift_error)
    elif drift_report_df is None or drift_report_df.empty:
        st.info("No drift output available.")
    else:
        drift_count = int((drift_report_df["status"] == "Drift").sum())

        if drift_count > 0:
            st.error(f"{drift_count} features showing drift - consider retraining")
        else:
            st.success("No features currently flagged as Drift.")

        def style_status_row(row: pd.Series) -> list[str]:
            colour_map = {
                "OK": "#e8f5e9",
                "Warning": "#fff8e1",
                "Drift": "#ffebee",
            }
            bg = colour_map.get(str(row.get("status", "")), "#ffffff")
            return [f"background-color: {bg}" for _ in row]

        st.dataframe(
            drift_report_df.style.apply(style_status_row, axis=1),
            use_container_width=True,
            hide_index=True,
        )

        psi_df = drift_report_df[drift_report_df["type"] == "numeric"].copy()
        psi_df["statistic"] = pd.to_numeric(psi_df["statistic"], errors="coerce")
        psi_df = psi_df.dropna(subset=["statistic"])

        if not psi_df.empty:
            fig_psi = go.Figure(
                go.Bar(
                    x=psi_df["feature"],
                    y=psi_df["statistic"],
                    marker_color="#4c78a8",
                    name="PSI",
                )
            )
            fig_psi.add_hline(y=0.1, line_dash="dash", line_color="#f39c12", annotation_text="Warning 0.1")
            fig_psi.add_hline(y=0.2, line_dash="dash", line_color="#e74c3c", annotation_text="Drift 0.2")
            fig_psi.update_layout(
                title="PSI by Numerical Feature",
                xaxis_title="Feature",
                yaxis_title="PSI",
                height=420,
                template=PLOTLY_TEMPLATE,
            )
            st.plotly_chart(fig_psi, use_container_width=True)
        else:
            st.info("No numerical PSI values available to plot.")
