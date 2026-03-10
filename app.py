"""
E-Commerce Churn Prediction Dashboard — Enhanced UI
LightGBM deployment version
"""

import json
from pathlib import Path

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

# -- page config --
st.set_page_config(
    page_title="E-Commerce Churn Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- custom CSS --
st.markdown("""
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
""", unsafe_allow_html=True)

# -- file path constants --
BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "lgbm_model.joblib"
NUM_IMPUTER_PATH = BASE / "lgbm_num_imputer.joblib"
CAT_IMPUTER_PATH = BASE / "lgbm_cat_imputer.joblib"
ENCODER_PATH = BASE / "lgbm_encoder.joblib"
SCALER_PATH = BASE / "lgbm_scaler.joblib"
META_PATH = BASE / "lgbm_metadata.json"
METRICS_ALL_MODELS_PATH = BASE / "metrics_all_models.csv"
CONFUSION_ALL_MODELS_PATH = BASE / "confusion_all_models.csv"
FEATURE_IMPORTANCE_PATH = BASE / "feature_importance.csv"

DEFAULT_ID_CANDIDATES = ["CustomerID", "customerID", "customer_id", "id", "ID"]

RAW_FEATURE_COLUMNS = [
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

RISK_COLOURS = {
    "Low": "#2ecc71",
    "Medium": "#f39c12",
    "High": "#e74c3c",
    "Critical": "#8e44ad",
}
PLOTLY_TEMPLATE = "plotly_white"


# ===== HELPER FUNCTIONS =====

def safe_read_csv(path):
    return pd.read_csv(path) if path.exists() else None


def to_binary_series(s):
    s = s.copy()
    if s.dtype == object:
        mapping = {"Yes": 1, "No": 0, "yes": 1, "no": 0, "1": 1, "0": 0}
        s = s.map(mapping).fillna(0).astype(int)
    return s.astype(int)


def pick_customer_id_column(df):
    for c in DEFAULT_ID_CANDIDATES:
        if c in df.columns:
            return c
    for c in df.columns:
        if "customer" in c.lower() and "id" in c.lower():
            return c
    return None


def risk_tier_from_proba(proba):
    return pd.cut(
        proba,
        bins=[0, 0.25, 0.50, 0.75, 1.01],
        labels=["Low", "Medium", "High", "Critical"],
        include_lowest=True,
    )


def safe_div(a, b, fill=0):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.where(np.abs(b) > 1e-12, a / b, fill)


def simple_action_recommendations(row):
    actions = []
    if row.get("Complain", 0) == 1:
        actions.append("Follow up on recent complaint — prioritise complaint resolution within 48 hours.")
    if row.get("SatisfactionScore", 5) <= 2:
        actions.append("Launch satisfaction-recovery outreach — personal call or targeted survey.")
    if row.get("CashbackAmount", 999) < 130:
        actions.append("Offer a cashback boost — customers with low cashback tend to churn more.")
    if row.get("CouponUsed", 999) < 1:
        actions.append("Send a coupon incentive — encourage first-time or repeat coupon usage.")
    if row.get("DaySinceLastOrder", 0) > 10:
        actions.append("Trigger a win-back campaign — the customer has not ordered recently.")
    if not actions:
        actions.append("No immediate risk signals detected — continue standard engagement.")
    return actions


# -- feature engineering --

def add_feature_engineering(df):
    df_fe = df.copy()

    # Ensure expected numeric columns are numeric when present
    numeric_hint_cols = [
        "Tenure", "CityTier", "WarehouseToHome", "HourSpendOnApp",
        "NumberOfDeviceRegistered", "SatisfactionScore", "NumberOfAddress",
        "Complain", "OrderAmountHikeFromlastYear", "CouponUsed",
        "OrderCount", "DaySinceLastOrder", "CashbackAmount"
    ]
    for c in numeric_hint_cols:
        if c in df_fe.columns:
            df_fe[c] = pd.to_numeric(df_fe[c], errors="coerce")

    # BLOCK 1: RFM-STYLE FEATURES
    df_fe["Recency_Score"] = safe_div(df_fe["DaySinceLastOrder"], df_fe["Tenure"] + 1)
    df_fe["Order_Frequency"] = safe_div(df_fe["OrderCount"], df_fe["Tenure"] + 1)
    df_fe["Cashback_per_Order"] = safe_div(df_fe["CashbackAmount"], df_fe["OrderCount"] + 1)
    df_fe["Coupon_per_Order"] = safe_div(df_fe["CouponUsed"], df_fe["OrderCount"] + 1)
    df_fe["AvgHike_per_Order"] = safe_div(df_fe["OrderAmountHikeFromlastYear"], df_fe["OrderCount"] + 1)
    df_fe["Recency_Bin"] = pd.cut(
        df_fe["DaySinceLastOrder"], bins=[-1, 5, 15, 30, 9999], labels=[0, 1, 2, 3]
    ).astype(float)

    # BLOCK 2: TENURE / LIFECYCLE FEATURES
    df_fe["Lifecycle_Stage"] = pd.cut(
        df_fe["Tenure"], bins=[-1, 1, 6, 12, 9999], labels=[0, 1, 2, 3]
    ).astype(float)
    df_fe["Tenure_sq"] = df_fe["Tenure"] ** 2
    df_fe["App_Hours_per_Month"] = safe_div(df_fe["HourSpendOnApp"], df_fe["Tenure"] + 1)
    df_fe["Address_per_Month"] = safe_div(df_fe["NumberOfAddress"], df_fe["Tenure"] + 1)
    df_fe["Device_per_Month"] = safe_div(df_fe["NumberOfDeviceRegistered"], df_fe["Tenure"] + 1)

    # BLOCK 3: SATISFACTION & COMPLAINT FEATURES
    df_fe["Inv_Satisfaction"] = 6 - df_fe["SatisfactionScore"]
    df_fe["Satisfaction_sq"] = df_fe["SatisfactionScore"] ** 2
    df_fe["Complaint_Severity"] = df_fe["Complain"] * df_fe["Inv_Satisfaction"]
    df_fe["Satisfaction_Decay"] = safe_div(df_fe["Inv_Satisfaction"], df_fe["Tenure"] + 1)
    df_fe["Chronic_Dissatisfaction"] = (
        (df_fe["Complain"] == 1) & (df_fe["SatisfactionScore"] <= 2)
    ).astype(int)
    df_fe["High_Risk_Flag"] = (
        (df_fe["Complain"] == 1) & (df_fe["SatisfactionScore"] <= 3)
    ).astype(int)

    # BLOCK 4: ENGAGEMENT COMPOSITE FEATURES
    _h = df_fe["HourSpendOnApp"] / (df_fe["HourSpendOnApp"].max() + 1e-9)
    _oc = df_fe["OrderCount"] / (df_fe["OrderCount"].max() + 1e-9)
    _cu = df_fe["CouponUsed"] / (df_fe["CouponUsed"].max() + 1e-9)
    df_fe["Engagement_Score"] = (_h + _oc + _cu) / 3.0 * 100
    df_fe["Low_Engagement_Flag"] = (df_fe["Order_Frequency"] < 1).astype(int)
    df_fe["Tenure_x_OrderCount"] = df_fe["Tenure"] * df_fe["OrderCount"]
    df_fe["App_x_Coupon"] = df_fe["HourSpendOnApp"] * df_fe["CouponUsed"]
    df_fe["Satisfaction_x_Orders"] = df_fe["SatisfactionScore"] * df_fe["OrderCount"]

    # BLOCK 5: DOMAIN-ENCODED RISK SCORES
    _pay_risk = {
        "Cash on Delivery": 3, "COD": 3, "E wallet": 2, "UPI": 2,
        "Debit Card": 1, "Credit Card": 1, "CC": 1,
    }
    df_fe["Payment_Risk"] = df_fe["PreferredPaymentMode"].map(_pay_risk).fillna(2).astype(int)

    _cat_risk = {
        "Mobile": 3, "Mobile Phone": 3, "Laptop & Accessory": 2,
        "Others": 2, "Fashion": 1, "Grocery": 1,
    }
    df_fe["Category_Risk"] = df_fe["PreferedOrderCat"].map(_cat_risk).fillna(2).astype(int)
    df_fe["CityTier_Risk"] = df_fe["CityTier"].map({1: 1, 2: 2, 3: 3}).fillna(2).astype(int)

    # BLOCK 6: DISTANCE & DELIVERY FEATURES
    df_fe["Log_WarehouseToHome"] = np.log1p(df_fe["WarehouseToHome"].clip(lower=0))
    df_fe["Far_Warehouse"] = (
        df_fe["WarehouseToHome"] > df_fe["WarehouseToHome"].quantile(0.75)
    ).astype(int)
    df_fe["Distance_x_Dissatisfaction"] = df_fe["WarehouseToHome"] * df_fe["Inv_Satisfaction"]

    # BLOCK 7: LOG TRANSFORMS
    for c in [
        "CashbackAmount",
        "OrderAmountHikeFromlastYear",
        "DaySinceLastOrder",
        "NumberOfAddress",
        "CouponUsed",
        "OrderCount",
    ]:
        if c in df_fe.columns:
            df_fe["Log_" + c] = np.log1p(df_fe[c].clip(lower=0))

    # BLOCK 8: POLYNOMIAL (SQUARED) FEATURES
    for c in ["DaySinceLastOrder", "CashbackAmount", "WarehouseToHome"]:
        if c in df_fe.columns:
            df_fe[c + "_sq"] = df_fe[c] ** 2

    # BLOCK 9: CASHBACK TIER
    df_fe["Cashback_Tier"] = pd.cut(
        df_fe["CashbackAmount"],
        bins=[-1, 100, 175, 250, 99999],
        labels=["Low", "Medium", "High", "Premium"],
    ).astype(str)

    # BLOCK 10: COMPOSITE CHURN RISK SCORE
    dsl_norm = df_fe["DaySinceLastOrder"] / (df_fe["DaySinceLastOrder"].max() + 1e-9)
    ten_norm = 1 - df_fe["Tenure"] / (df_fe["Tenure"].max() + 1e-9)
    cb_norm = 1 - df_fe["CashbackAmount"] / (df_fe["CashbackAmount"].max() + 1e-9)
    df_fe["Composite_Risk_Score"] = (
        df_fe["Complain"] * 3.0
        + df_fe["Inv_Satisfaction"]
        + dsl_norm
        + ten_norm
        + cb_norm
    )
    return df_fe


# -- preprocessing (mirrors notebook) --

def preprocess_like_notebook(df_model_input, num_imputer, cat_imputer, encoder, scaler, num_cols, cat_cols):
    df_proc = df_model_input.copy()

    num_cols_present = [c for c in num_cols if c in df_proc.columns]
    cat_cols_present = [c for c in cat_cols if c in df_proc.columns]

    for c in num_cols_present:
        df_proc[c] = pd.to_numeric(df_proc[c], errors="coerce")

    for c in cat_cols_present:
        df_proc[c] = df_proc[c].astype(object)

    if num_cols_present:
        df_proc.loc[:, num_cols_present] = num_imputer.transform(df_proc[num_cols_present])
        for c in num_cols_present:
            df_proc[c] = pd.to_numeric(df_proc[c], errors="coerce")

    if cat_cols_present:
        cat_array = cat_imputer.transform(df_proc[cat_cols_present])
        cat_df = pd.DataFrame(cat_array, columns=cat_cols_present, index=df_proc.index)

        for c in cat_cols_present:
            cat_df[c] = cat_df[c].astype(str)

        encoded = encoder.transform(cat_df[cat_cols_present])
        ohe_cols = encoder.get_feature_names_out(cat_cols_present)
        df_proc_num = df_proc.drop(columns=cat_cols_present).reset_index(drop=True)
        encoded_df = pd.DataFrame(encoded, columns=ohe_cols, index=df_proc_num.index)
        df_proc = pd.concat([df_proc_num, encoded_df], axis=1)

    # Keep feature order consistent for scaler
    if hasattr(scaler, "feature_names_in_"):
        scale_cols = [c for c in scaler.feature_names_in_ if c in df_proc.columns]
    else:
        scale_cols = df_proc.select_dtypes(include=[np.number]).columns.tolist()

    if scale_cols:
        df_proc.loc[:, scale_cols] = scaler.transform(df_proc[scale_cols])

    # Final safety: cast all to numeric
    for c in df_proc.columns:
        df_proc[c] = pd.to_numeric(df_proc[c], errors="coerce")

    return df_proc


# ===== HEADER =====

st.markdown("""
# E-Commerce Churn Prediction Dashboard
**Identify at-risk customers, understand churn drivers, and take targeted retention actions.**
""")

# ===== SIDEBAR =====

with st.sidebar:
    st.markdown("## Dashboard Controls")
    st.markdown("---")

    threshold = st.slider(
        "Churn Probability Threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.01,
        help="Customers with predicted probability above this value are classified as likely to churn.",
    )

    st.markdown("---")
    uploaded = st.file_uploader(
        "Upload Customer CSV",
        type=["csv"],
        help="Upload a raw or feature-engineered CSV. If it contains a Churn column, evaluation metrics will be computed automatically.",
    )

    st.markdown("---")
    with st.expander("Artifact Status", expanded=False):
        artifact_checks = {
            "LightGBM Model": MODEL_PATH.exists(),
            "Numerical Imputer": NUM_IMPUTER_PATH.exists(),
            "Categorical Imputer": CAT_IMPUTER_PATH.exists(),
            "Encoder": ENCODER_PATH.exists(),
            "Scaler": SCALER_PATH.exists(),
            "Metadata": META_PATH.exists(),
            "Feature Importance": FEATURE_IMPORTANCE_PATH.exists(),
        }
        for name, ok in artifact_checks.items():
            icon = "Available" if ok else "Missing"
            st.markdown("**{}:** {}".format(name, icon))

# ===== LOAD ARTIFACTS =====

@st.cache_resource
def load_artifacts():
    return (
        joblib.load(NUM_IMPUTER_PATH),
        joblib.load(CAT_IMPUTER_PATH),
        joblib.load(ENCODER_PATH),
        joblib.load(SCALER_PATH),
        joblib.load(MODEL_PATH),
        json.loads(META_PATH.read_text(encoding="utf-8")),
    )


try:
    num_imputer, cat_imputer, encoder, scaler, model, meta = load_artifacts()
except Exception as e:
    st.error("Failed to load model artifacts. Ensure all LightGBM .joblib and .json files are in the same folder as app.py.")
    st.exception(e)
    st.stop()

target_col = meta.get("target", "Churn")
raw_input_columns = meta.get("raw_input_columns", [])
num_cols = meta.get("num_cols", [])
cat_cols = meta.get("cat_cols", [])

metrics_all = safe_read_csv(METRICS_ALL_MODELS_PATH)
conf_all = safe_read_csv(CONFUSION_ALL_MODELS_PATH)
feat_imp_external = safe_read_csv(FEATURE_IMPORTANCE_PATH)

# ===== LANDING (no upload) =====

if uploaded is None:
    st.markdown("---")
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("""
        #### Getting Started
        1. Upload a customer CSV using the sidebar.  
        2. The model will score each customer automatically.  
        3. Explore the interactive tabs for insights.
        """)

    with col_b:
        st.markdown("""
        #### Expected Columns
        Your CSV should contain some or all of these raw features for best results:
        """)
        st.code(", ".join(RAW_FEATURE_COLUMNS[:9]), language=None)
        st.code(", ".join(RAW_FEATURE_COLUMNS[9:]), language=None)

    with col_c:
        st.markdown("""
        #### Tips
        - Include a **Churn** column for evaluation metrics.  
        - Include a **CustomerID** column for individual lookup.  
        - Feature engineering is applied automatically.
        """)

    st.stop()

# ===== READ & PREPARE UPLOAD =====

try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error("Could not read the uploaded CSV.")
    st.exception(e)
    st.stop()

has_target = target_col in df.columns
y_true = to_binary_series(df[target_col].copy()) if has_target else None

df_original = df.copy()
df_features = df.drop(columns=[target_col]).copy() if has_target else df.copy()

required_for_fe = [c for c in RAW_FEATURE_COLUMNS if c in df_features.columns]
if len(required_for_fe) == len(RAW_FEATURE_COLUMNS):
    df_features = add_feature_engineering(df_features)

id_col_detected = pick_customer_id_column(df_features)
if id_col_detected and id_col_detected in df_features.columns:
    df_model_input = df_features.drop(columns=[id_col_detected]).copy()
else:
    df_model_input = df_features.copy()

missing_for_model = [c for c in raw_input_columns if c not in df_model_input.columns]
if raw_input_columns and missing_for_model:
    st.error("Missing required feature columns:")
    st.write(missing_for_model)
    st.stop()

if raw_input_columns:
    df_model_input = df_model_input[raw_input_columns]

# ===== PREDICT =====

try:
    X_model = preprocess_like_notebook(
        df_model_input,
        num_imputer,
        cat_imputer,
        encoder,
        scaler,
        num_cols,
        cat_cols,
    )
    proba = model.predict_proba(X_model)[:, 1]
    pred = (proba >= threshold).astype(int)
except Exception as e:
    st.error("Prediction failed.")
    st.exception(e)
    st.stop()

out = df_original.copy()
eng_cols_to_add = [c for c in df_features.columns if c not in out.columns]
if eng_cols_to_add:
    out = pd.concat(
        [out.reset_index(drop=True), df_features[eng_cols_to_add].reset_index(drop=True)],
        axis=1,
    )

out["Churn_probability"] = proba
out["Churn_pred"] = pred
out["Risk_Tier"] = risk_tier_from_proba(proba)

# ===== FILTERS =====

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
        st.caption("No standard filter columns found in this upload.")
        filtered = out.copy()
    else:
        cols = st.columns(min(3, len(available_filters)))
        selected = {}
        for i, c in enumerate(available_filters):
            with cols[i % len(cols)]:
                opts = sorted([x for x in out[c].dropna().unique().tolist()])
                sel = st.multiselect("{}".format(c), options=opts, default=opts)
                selected[c] = set(sel)

        mask = np.ones(len(out), dtype=bool)
        for c, keep in selected.items():
            mask &= out[c].isin(list(keep))

        filtered = out.loc[mask].copy()

    st.info("Showing {:,} of {:,} customers after filters.".format(len(filtered), len(out)))

# ===== KPI CARDS =====

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
    meta_metrics = meta.get("metrics", {})
    k3.metric("Model Accuracy", "{:.3f}".format(meta_metrics.get("accuracy", 0)))
    k4.metric("Recall", "{:.3f}".format(meta_metrics.get("recall", 0)))

st.markdown("---")

# ===== TABS =====

tab_model, tab_drivers, tab_segments, tab_lookup, tab_data, tab_conclusion = st.tabs([
    "Model Performance",
    "Top Churn Drivers",
    "Risk Segmentation",
    "Customer Lookup",
    "Scored Data",
    "Model Conclusions",
])

# -- TAB 1: MODEL PERFORMANCE --
with tab_model:
    st.markdown("### LightGBM Performance")

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

        st.markdown("")

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
                        name="LightGBM (AUC = {:.4f})".format(roc),
                        line=dict(color="#667eea", width=3),
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
        st.info("Upload a CSV with a Churn column to view evaluation metrics and charts.")

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

    if conf_all is not None and "Model" in conf_all.columns:
        st.markdown("---")
        st.markdown("### Confusion Matrix Comparison")
        models = conf_all["Model"].unique().tolist()
        chosen_model = st.selectbox("Select a model to inspect", models)
        row_cm = conf_all.loc[conf_all["Model"] == chosen_model].iloc[0]
        needed = ["TN", "FP", "FN", "TP"]

        if all(k in conf_all.columns for k in needed):
            cm_vals = np.array([
                [int(row_cm["TN"]), int(row_cm["FP"])],
                [int(row_cm["FN"]), int(row_cm["TP"])]
            ])
            labels = ["Stayed", "Churned"]
            fig_cm2 = go.Figure(
                data=go.Heatmap(
                    z=cm_vals[::-1],
                    x=labels,
                    y=labels[::-1],
                    text=cm_vals[::-1],
                    texttemplate="%{text}",
                    textfont=dict(size=18),
                    colorscale="Oranges",
                    showscale=False,
                )
            )
            fig_cm2.update_layout(
                title="Confusion Matrix - {}".format(chosen_model),
                xaxis_title="Predicted",
                yaxis_title="Actual",
                height=380,
                template=PLOTLY_TEMPLATE,
            )
            st.plotly_chart(fig_cm2, use_container_width=True)

# -- TAB 2: TOP DRIVERS / SHAP --
with tab_drivers:
    st.markdown("### Actionable Churn Drivers")
    st.caption("Understanding why customers churn enables targeted retention strategies.")

    guide_data = [
        ["Recency and Activity", "DaySinceLastOrder, Recency_Score", "Long gaps since last order strongly signal churn."],
        ["Complaints", "Complain, Complaint_Severity", "Dissatisfied customers are far more likely to leave."],
        ["Tenure", "Tenure, Lifecycle_Stage", "New customers churn more; long-tenured customers are stickier."],
        ["Order Behaviour", "OrderCount, Order_Frequency", "Low purchase engagement increases churn risk."],
        ["Financial", "CashbackAmount, Cashback_per_Order", "Incentives influence retention and repeat purchasing."],
        ["Satisfaction", "SatisfactionScore, Inv_Satisfaction", "Direct customer sentiment signal."],
        ["Composite Risk", "Composite_Risk_Score, High_Risk_Flag", "Engineered risk features support alerting."],
        ["Logistics", "WarehouseToHome, Far_Warehouse", "Distance affects delivery experience and churn."],
    ]
    guide = pd.DataFrame(guide_data, columns=["Category", "Key Features", "Why It Matters"])
    st.dataframe(guide, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### SHAP Feature Importance")

    if feat_imp_external is not None and {"feature", "importance"}.issubset(feat_imp_external.columns):
        top_n = st.slider("Number of top features to display", 5, 40, 20, key="shap_top_n")
        fi = feat_imp_external.sort_values("importance", ascending=False).head(top_n)

        fig_shap = go.Figure(
            go.Bar(
                x=fi["importance"].values[::-1],
                y=fi["feature"].values[::-1],
                orientation="h",
                marker=dict(
                    color=fi["importance"].values[::-1],
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="SHAP"),
                ),
            )
        )
        fig_shap.update_layout(
            title="Top {} Features by Mean Absolute SHAP Value".format(top_n),
            xaxis_title="Mean Absolute SHAP Value",
            height=max(400, top_n * 24),
            template=PLOTLY_TEMPLATE,
            margin=dict(l=200),
        )
        st.plotly_chart(fig_shap, use_container_width=True)
    else:
        st.warning("feature_importance.csv not found. Export it from the notebook to enable this chart.")

# -- TAB 3: RISK SEGMENTATION --
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
        lc_col = None
        if "Lifecycle_Stage_Label" in filtered.columns:
            lc_col = "Lifecycle_Stage_Label"
        elif "Lifecycle_Stage" in filtered.columns:
            lc_col = "Lifecycle_Stage"

        if lc_col:
            lc_counts = filtered[lc_col].value_counts()
            fig_lc = go.Figure(
                go.Pie(
                    labels=lc_counts.index.astype(str).tolist(),
                    values=lc_counts.values.tolist(),
                    hole=0.45,
                    textinfo="label+percent",
                    textfont=dict(size=14),
                )
            )
            fig_lc.update_layout(title="Lifecycle Stage Breakdown", height=420, template=PLOTLY_TEMPLATE)
            st.plotly_chart(fig_lc, use_container_width=True)
        else:
            st.info("Lifecycle stage column not found.")

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

    st.markdown("---")

    st.markdown("#### Cashback Tier vs Churn Rate")
    if "Cashback_Tier" in filtered.columns and has_target and target_col in filtered.columns:
        tmp = filtered.copy()
        tmp[target_col] = to_binary_series(tmp[target_col])
        grp = tmp.groupby("Cashback_Tier", observed=False)[target_col].mean().reset_index()
        grp.columns = ["Cashback_Tier", "Churn_Rate"]

        fig_cb = px.bar(
            grp,
            x="Cashback_Tier",
            y="Churn_Rate",
            text=grp["Churn_Rate"].map("{:.1%}".format),
            color="Churn_Rate",
            color_continuous_scale="Reds",
            labels={"Churn_Rate": "Churn Rate"},
        )
        fig_cb.update_layout(
            title="Churn Rate by Cashback Tier (Actual)",
            yaxis_tickformat=".0%",
            height=380,
            template=PLOTLY_TEMPLATE,
            showlegend=False,
        )
        fig_cb.update_traces(textposition="outside")
        st.plotly_chart(fig_cb, use_container_width=True)
    elif "Cashback_Tier" not in filtered.columns:
        st.info("Cashback_Tier not available.")
    else:
        st.info("Upload must include a Churn column to compute churn rate by Cashback Tier.")

# -- TAB 4: INDIVIDUAL LOOKUP --
with tab_lookup:
    st.markdown("### Individual Customer Lookup")

    id_col = pick_customer_id_column(filtered)
    if id_col is None:
        st.warning("No CustomerID column found. Add one to enable individual lookup.")
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

            h1.markdown("""
            <div style="text-align:center; padding:20px; border-radius:12px; background:{}15; border:2px solid {};">
                <div style="font-size:0.85rem; color:#666; font-weight:600;">CHURN PROBABILITY</div>
                <div style="font-size:2.2rem; font-weight:700; color:{};">{:.1%}</div>
            </div>
            """.format(prob_colour, prob_colour, prob_colour, prob), unsafe_allow_html=True)

            tier_colour = RISK_COLOURS.get(tier, "#999")
            h2.markdown("""
            <div style="text-align:center; padding:20px; border-radius:12px; background:{}15; border:2px solid {};">
                <div style="font-size:0.85rem; color:#666; font-weight:600;">RISK TIER</div>
                <div style="font-size:2.2rem; font-weight:700; color:{};">{}</div>
            </div>
            """.format(tier_colour, tier_colour, tier_colour, tier), unsafe_allow_html=True)

            pred_colour = "#e74c3c" if pred_label == 1 else "#2ecc71"
            pred_text = "Will Churn" if pred_label == 1 else "Will Stay"
            h3.markdown("""
            <div style="text-align:center; padding:20px; border-radius:12px; background:{}15; border:2px solid {};">
                <div style="font-size:0.85rem; color:#666; font-weight:600;">PREDICTION</div>
                <div style="font-size:2.2rem; font-weight:700; color:{};">{}</div>
            </div>
            """.format(pred_colour, pred_colour, pred_colour, pred_text), unsafe_allow_html=True)

            st.markdown("")

            col_snap, col_action = st.columns([3, 2])

            with col_snap:
                st.markdown("#### Customer Profile")
                driver_candidates = [
                    "DaySinceLastOrder",
                    "Recency_Score",
                    "Complain",
                    "Complaint_Severity",
                    "Tenure",
                    "Lifecycle_Stage",
                    "OrderCount",
                    "Order_Frequency",
                    "CouponUsed",
                    "CashbackAmount",
                    "Cashback_Tier",
                    "SatisfactionScore",
                    "Composite_Risk_Score",
                    "High_Risk_Flag",
                    "Engagement_Score",
                    "WarehouseToHome",
                ]
                snapshot_cols = [c for c in driver_candidates if c in row.index]
                if snapshot_cols:
                    snapshot = pd.DataFrame({
                        "Feature": snapshot_cols,
                        "Value": [row[c] for c in snapshot_cols]
                    })
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

# -- TAB 5: SCORED DATA --
with tab_data:
    st.markdown("### Scored Customer Data")

    s1, s2, s3 = st.columns(3)
    s1.metric("Total Rows", "{:,}".format(len(filtered)))
    s2.metric("Predicted Churners", "{:,}".format(int(filtered["Churn_pred"].sum())))
    s3.metric("Average Probability", "{:.2%}".format(filtered["Churn_probability"].mean()))

    st.markdown("")

    show_n = st.slider(
        "Rows to preview",
        10,
        min(500, len(filtered)),
        min(100, len(filtered)),
        key="data_rows",
    )
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
        color_discrete_sequence=["#667eea"],
        labels={"Churn_probability": "Churn Probability"},
    )
    fig_hist.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="red",
        annotation_text="Threshold ({})".format(threshold)
    )
    fig_hist.update_layout(
        title="Distribution of Predicted Churn Probabilities",
        yaxis_title="Count",
        height=380,
        template=PLOTLY_TEMPLATE,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    if has_target and target_col in filtered.columns:
        actual_rate = to_binary_series(filtered[target_col]).mean() * 100
        st.info("Actual churn rate in filtered segment: {:.2f}%".format(actual_rate))

# -- TAB 6: MODEL CONCLUSIONS --
with tab_conclusion:
    st.markdown("### Model Conclusions & How Each Model Works")
    st.markdown("---")

    model_summary = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Random Forest",
            "XGBoost",
            "DNN (MLP)",
            "LightGBM (Full)",
            "LightGBM (Reduced)",
            "LSTM",
            "SVM (Baseline)",
            "SVM (Tuned)",
            "CatBoost",
        ],
        "F1 Score": [None, None, None, None, 0.97, 0.87, 0.45, None, 0.00, 0.905],
        "ROC AUC": [None, None, None, None, 0.999, 0.99, 0.72, 0.50, None, 0.996],
        "Verdict": [
            "Baseline",
            "Strong",
            "Strong",
            "Competitive",
            "Best overall",
            "Good (no risk features)",
            "Poor fit",
            "Poor fit",
            "Poor fit",
            "Very strong",
        ],
    })

    st.dataframe(model_summary, use_container_width=True, hide_index=True)

    plot_df = model_summary.dropna(subset=["F1 Score", "ROC AUC"])
    if not plot_df.empty:
        fig_conc = go.Figure()
        fig_conc.add_trace(go.Bar(
            x=plot_df["Model"], y=plot_df["F1 Score"],
            name="F1 Score", marker_color="#667eea",
        ))
        fig_conc.add_trace(go.Bar(
            x=plot_df["Model"], y=plot_df["ROC AUC"],
            name="ROC AUC", marker_color="#f39c12",
        ))
        fig_conc.update_layout(
            barmode="group",
            title="Model Performance Comparison — F1 Score vs ROC AUC",
            yaxis_title="Score",
            height=420,
            template=PLOTLY_TEMPLATE,
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig_conc, use_container_width=True)

    st.markdown("---")

    conclusions = [
        {
            "name": "Logistic Regression",
            "how": (
                "Logistic Regression fits a linear decision boundary by learning a weighted sum of features "
                "passed through a sigmoid function. It estimates the probability that a customer churns based on "
                "a linear combination of the input features."
            ),
            "result": (
                "Serves as the baseline model. It is fast and interpretable but cannot capture the complex, "
                "non-linear feature interactions critical for accurate churn prediction."
            ),
        },
        {
            "name": "Random Forest",
            "how": (
                "Random Forest builds an ensemble of many decision trees, each trained on a random subset of "
                "data and features. Predictions are made by majority vote across all trees."
            ),
            "result": (
                "Performed strongly and offered good interpretability through feature importance, though it can "
                "be slower than boosting models at inference."
            ),
        },
        {
            "name": "XGBoost",
            "how": (
                "XGBoost is a sequential gradient boosting algorithm that builds trees one at a time, where each "
                "new tree corrects the errors of the previous ensemble."
            ),
            "result": (
                "Delivered excellent results, but this dashboard now deploys LightGBM instead."
            ),
        },
        {
            "name": "DNN (Deep Neural Network / MLP)",
            "how": (
                "The DNN uses a multi-layer perceptron architecture with dense layers and dropout for "
                "regularisation. It learns non-linear feature representations through back-propagation."
            ),
            "result": (
                "Showed competitive performance but is more complex to configure and less interpretable for "
                "business stakeholders."
            ),
        },
        {
            "name": "LightGBM (Full Features)",
            "how": (
                "LightGBM uses a leaf-wise tree growth strategy, making it efficient and often highly accurate "
                "for structured tabular datasets. It uses histogram-based splitting for speed."
            ),
            "result": (
                "This is the deployed model in the dashboard. It achieved the strongest overall performance and "
                "is well suited for churn prediction with engineered behavioural features."
            ),
        },
        {
            "name": "LightGBM (Reduced Features)",
            "how": (
                "This uses the same LightGBM algorithm but with a reduced feature set to test performance without "
                "certain domain-engineered risk variables."
            ),
            "result": (
                "Still performed very well, showing that core behaviour and engagement variables carry strong "
                "predictive power even without all composite risk features."
            ),
        },
        {
            "name": "LSTM",
            "how": (
                "LSTM is a recurrent neural network designed for sequential data with time-dependent structure."
            ),
            "result": (
                "It is not well suited for this flat tabular churn dataset and underperformed relative to tree-based models."
            ),
        },
        {
            "name": "SVM (Support Vector Machine)",
            "how": (
                "SVM finds a hyperplane that maximises the margin between classes and can model non-linear "
                "boundaries through kernels."
            ),
            "result": (
                "It struggled on this high-dimensional one-hot encoded churn problem and is not recommended here."
            ),
        },
        {
            "name": "CatBoost",
            "how": (
                "CatBoost uses ordered boosting and symmetric trees and handles categorical variables very well."
            ),
            "result": (
                "A very strong alternative to LightGBM, especially when minimal categorical preprocessing is preferred."
            ),
        },
    ]

    for c in conclusions:
        with st.expander(c["name"], expanded=False):
            st.markdown("**How it works:** {}".format(c["how"]))
            st.markdown("**Result & conclusion:** {}".format(c["result"]))

    st.markdown("---")
    st.markdown("### Final Recommendation")

    rec_data = pd.DataFrame({
        "Criterion": [
            "Best overall performance",
            "Currently deployed in dashboard",
            "Best balance of simplicity & power",
            "Best interpretability",
            "Not recommended for this data",
        ],
        "Recommended Model": [
            "LightGBM (Full Features)",
            "LightGBM",
            "CatBoost",
            "Logistic Regression / Random Forest",
            "LSTM, SVM",
        ],
    })
    st.table(rec_data)

    st.success(
        "LightGBM is the deployed model in this dashboard and should be used for production scoring. "
        "It provides strong predictive performance on this tabular churn dataset. "
        "CatBoost remains a strong alternative, while LSTM and SVM are not competitive for this use case."
    )