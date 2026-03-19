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

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
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
FEATURE_IMPORTANCE_PATH = BASE / "feature_importance.csv"

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


@st.cache_data(ttl=60)
def fetch_firestore_customers(collection_name: str = CURRENT_COLLECTION) -> pd.DataFrame:
    db = get_firestore_client()
    docs = db.collection(collection_name).stream()

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


# =========================
# HEADER
# =========================
st.markdown(
    """
    # E-Commerce Churn Prediction Dashboard
    **Live churn scoring using the weighted ensemble from `ensemble.ipynb`.**
    """
)

# =========================
# SIDEBAR
# =========================
with st.sidebar:
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

    uploaded = None
    if data_source == "Upload CSV":
        st.markdown("---")
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
        }
        for name, ok in artifact_checks.items():
            st.markdown("**{}:** {}".format(name, "Available" if ok else "Missing"))

# =========================
# LOAD MODEL ARTIFACTS
# =========================
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
        df = fetch_firestore_customers(CURRENT_COLLECTION)
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

# =========================
# TABS
# =========================
tab_model, tab_drivers, tab_segments, tab_lookup, tab_data, tab_conclusion = st.tabs([
    "Model Performance",
    "Top Churn Drivers",
    "Risk Segmentation",
    "Customer Lookup",
    "Scored Data",
    "Model Conclusions",
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

    if feat_imp_external is not None and {"feature", "importance"}.issubset(feat_imp_external.columns):
        top_n = st.slider("Number of top features to display", 5, 40, 20, key="shap_top_n")
        fi = feat_imp_external.sort_values("importance", ascending=False).head(top_n)

        fig_shap = go.Figure(
            go.Bar(
                x=fi["importance"].values[::-1],
                y=fi["feature"].values[::-1],
                orientation="h",
            )
        )
        fig_shap.update_layout(
            title="Top {} Features".format(top_n),
            xaxis_title="Importance",
            height=max(400, top_n * 24),
            template=PLOTLY_TEMPLATE,
            margin=dict(l=200),
        )
        st.plotly_chart(fig_shap, use_container_width=True)
    else:
        st.warning("feature_importance.csv not found. Export it from the notebook to enable this chart.")

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
# TAB 6: MODEL CONCLUSIONS
# =========================
with tab_conclusion:
    st.markdown("### Model Conclusions")
    st.markdown("---")

    st.markdown(
        """
        The deployed dashboard uses a **weighted ensemble**.

        **Pipeline used in training and inference**
        - Drop ID columns such as `CustomerID`
        - Numeric preprocessing: median imputation, polynomial interaction features, standard scaling
        - Categorical preprocessing: most-frequent imputation, one-hot encoding
        - Feature selection: `SelectKBest(k=50)`
        - Models: Random Forest, XGBoost, LightGBM
        - Final prediction: weighted average of model probabilities using recall-derived weights
        """
    )

    conclusion_df = pd.DataFrame(
        {
            "Component": [
                "Random Forest",
                "XGBoost",
                "LightGBM",
                "Ensemble Strategy",
            ],
            "Role": [
                "Captures robust non-linear tree interactions",
                "Strong boosted learner with flexible decision boundaries",
                "Efficient boosted tree model for tabular data",
                "Combines all three using recall-normalised weights",
            ],
        }
    )
    st.dataframe(conclusion_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown(
        """
        **Why this setup is strong**
        
        Tree-based models are effective for tabular churn data because they capture non-linear patterns,
        interaction effects, and mixed numeric-categorical behaviour well. The weighted ensemble improves
        stability and balances strengths across the three models instead of relying on only one model.
        """
    )