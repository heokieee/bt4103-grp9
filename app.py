import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)

# =========================
# Page config + constants
# =========================
st.set_page_config(page_title="E-Commerce Churn Dashboard", layout="wide")

APP_DIR = Path(__file__).resolve().parent

MODEL_PATH = APP_DIR / "xgb_model.joblib"
NUM_IMPUTER_PATH = APP_DIR / "xgb_num_imputer.joblib"
CAT_IMPUTER_PATH = APP_DIR / "xgb_cat_imputer.joblib"
ENCODER_PATH = APP_DIR / "xgb_encoder.joblib"
SCALER_PATH = APP_DIR / "xgb_scaler.joblib"
META_PATH = APP_DIR / "xgb_metadata.json"

# Optional files
METRICS_ALL_MODELS_PATH = APP_DIR / "metrics_all_models.csv"
CONFUSION_ALL_MODELS_PATH = APP_DIR / "confusion_all_models.csv"
FEATURE_IMPORTANCE_PATH = APP_DIR / "feature_importance.csv"

DEFAULT_ID_CANDIDATES = [
    "CustomerID", "CustomerId", "customer_id", "Customer_ID",
    "ID", "id", "userid", "user_id"
]

RAW_FEATURE_COLUMNS = [
    "Tenure", "CityTier", "WarehouseToHome", "HourSpendOnApp",
    "NumberOfDeviceRegistered", "SatisfactionScore", "NumberOfAddress",
    "Complain", "OrderAmountHikeFromlastYear", "CouponUsed",
    "OrderCount", "DaySinceLastOrder", "CashbackAmount",
    "PreferredLoginDevice", "PreferredPaymentMode", "Gender",
    "PreferedOrderCat", "MaritalStatus"
]

# =========================
# Helpers
# =========================
def safe_read_csv(path: Path):
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception:
        return None
    return None

def to_binary_series(s: pd.Series) -> pd.Series:
    if s.dtype == "object":
        s = (
            s.astype(str).str.strip().str.lower()
            .map({
                "yes": 1, "no": 0, "true": 1, "false": 0,
                "1": 1, "0": 0
            })
        )
    return s.astype(int)

def pick_customer_id_column(df: pd.DataFrame) -> Optional[str]:
    for c in DEFAULT_ID_CANDIDATES:
        if c in df.columns:
            return c
    for c in df.columns:
        if "id" in c.lower():
            return c
    return None

def risk_tier_from_proba(p: np.ndarray):
    bins = [0.0, 0.25, 0.5, 0.75, 1.000001]
    labels = ["Low", "Medium", "High", "Critical"]
    return pd.cut(p, bins=bins, labels=labels, include_lowest=True)

def plot_bar_metrics(metrics_df: pd.DataFrame):
    dfm = metrics_df.copy()

    rename_map = {}
    for col in dfm.columns:
        key = col.lower().replace(" ", "").replace("_", "")
        if key in ["rocauc", "rocaucscore"]:
            rename_map[col] = "ROC_AUC"
    dfm = dfm.rename(columns=rename_map)

    cols_needed = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
    available = [c for c in cols_needed if c in dfm.columns]

    if "Model" not in dfm.columns or len(available) == 0:
        st.info("metrics_all_models.csv is missing required columns.")
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    models = dfm["Model"].tolist()
    metrics = available
    x = np.arange(len(models))
    width = 0.8 / max(1, len(metrics))

    for i, m in enumerate(metrics):
        vals = dfm[m].values
        ax.bar(x + i * width - 0.4 + width / 2, vals, width=width, label=m)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_title("Model Comparison (Metrics)")
    ax.legend(ncol=min(5, len(metrics)))
    st.pyplot(fig)

def plot_confusion_from_counts(tn, fp, fn, tp, title="Confusion Matrix"):
    cm = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(title)
    st.pyplot(fig)

def plot_hist(values, title, xlabel):
    fig, ax = plt.subplots()
    ax.hist(values, bins=30)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    st.pyplot(fig)

def plot_pie_counts(series: pd.Series, title: str):
    counts = series.value_counts(dropna=False)
    fig, ax = plt.subplots()
    ax.pie(
        counts.values,
        labels=[str(i) for i in counts.index],
        autopct="%1.1f%%",
        startangle=90
    )
    ax.set_title(title)
    st.pyplot(fig)

def simple_action_recommendations(row: pd.Series):
    recs = []

    if "Complain" in row.index and row.get("Complain", 0) == 1:
        recs.append("Follow up on complaint and offer resolution or service credit.")

    if "SatisfactionScore" in row.index:
        try:
            if float(row["SatisfactionScore"]) <= 2:
                recs.append("Offer service recovery and concierge support due to very low satisfaction.")
        except Exception:
            pass

    if "CashbackAmount" in row.index:
        try:
            if float(row["CashbackAmount"]) <= 0:
                recs.append("Offer targeted cashback to re-engage the customer.")
        except Exception:
            pass

    if "CouponUsed" in row.index:
        try:
            if int(row["CouponUsed"]) == 0:
                recs.append("Send a time-limited coupon to stimulate the next purchase.")
        except Exception:
            pass

    if "DaySinceLastOrder" in row.index:
        try:
            if float(row["DaySinceLastOrder"]) >= 30:
                recs.append("Run a win-back campaign because the inactivity gap is long.")
        except Exception:
            pass

    if not recs:
        recs.append("Nudge with personalised recommendations and a light incentive.")

    return recs

def safe_div(a, b, fill=0.0):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.where(b != 0, a / b, fill)

def add_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df_fe = df.copy()

    # Block 1 · RFM-style features
    df_fe["Recency_Score"] = safe_div(df_fe["DaySinceLastOrder"], df_fe["Tenure"] + 1)
    df_fe["Order_Frequency"] = safe_div(df_fe["OrderCount"], df_fe["Tenure"] + 1)
    df_fe["Cashback_per_Order"] = safe_div(df_fe["CashbackAmount"], df_fe["OrderCount"] + 1)
    df_fe["Coupon_per_Order"] = safe_div(df_fe["CouponUsed"], df_fe["OrderCount"] + 1)
    df_fe["AvgHike_per_Order"] = safe_div(
        df_fe["OrderAmountHikeFromlastYear"], df_fe["OrderCount"] + 1
    )
    df_fe["Recency_Bin"] = pd.cut(
        df_fe["DaySinceLastOrder"],
        bins=[-1, 5, 15, 30, 9999],
        labels=[0, 1, 2, 3]
    ).astype(float)

    # Block 2 · Tenure / lifecycle features
    df_fe["Lifecycle_Stage"] = pd.cut(
        df_fe["Tenure"],
        bins=[-1, 1, 6, 12, 9999],
        labels=[0, 1, 2, 3]
    ).astype(float)

    lifecycle_map = {
        0.0: "New",
        1.0: "Early",
        2.0: "Established",
        3.0: "Loyal"
    }
    df_fe["Lifecycle_Stage_Label"] = df_fe["Lifecycle_Stage"].map(lifecycle_map)

    df_fe["Tenure_sq"] = df_fe["Tenure"] ** 2
    df_fe["App_Hours_per_Month"] = safe_div(df_fe["HourSpendOnApp"], df_fe["Tenure"] + 1)
    df_fe["Address_per_Month"] = safe_div(df_fe["NumberOfAddress"], df_fe["Tenure"] + 1)
    df_fe["Device_per_Month"] = safe_div(df_fe["NumberOfDeviceRegistered"], df_fe["Tenure"] + 1)

    # Block 3 · Satisfaction & complaint features
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

    # Block 4 · Engagement composite features
    _h = df_fe["HourSpendOnApp"] / (df_fe["HourSpendOnApp"].max() + 1e-9)
    _oc = df_fe["OrderCount"] / (df_fe["OrderCount"].max() + 1e-9)
    _cu = df_fe["CouponUsed"] / (df_fe["CouponUsed"].max() + 1e-9)
    df_fe["Engagement_Score"] = (_h + _oc + _cu) / 3.0 * 100
    df_fe["Low_Engagement_Flag"] = (df_fe["Order_Frequency"] < 1).astype(int)
    df_fe["Tenure_x_OrderCount"] = df_fe["Tenure"] * df_fe["OrderCount"]
    df_fe["App_x_Coupon"] = df_fe["HourSpendOnApp"] * df_fe["CouponUsed"]
    df_fe["Satisfaction_x_Orders"] = df_fe["SatisfactionScore"] * df_fe["OrderCount"]

    # Block 5 · Domain-encoded risk scores
    pay_risk = {
        "Cash on Delivery": 3, "COD": 3,
        "E wallet": 2, "UPI": 2,
        "Debit Card": 1, "Credit Card": 1, "CC": 1,
    }
    df_fe["Payment_Risk"] = (
        df_fe["PreferredPaymentMode"].map(pay_risk).fillna(2).astype(int)
    )

    cat_risk = {
        "Mobile": 3, "Mobile Phone": 3,
        "Laptop & Accessory": 2, "Others": 2,
        "Fashion": 1, "Grocery": 1,
    }
    df_fe["Category_Risk"] = (
        df_fe["PreferedOrderCat"].map(cat_risk).fillna(2).astype(int)
    )

    df_fe["CityTier_Risk"] = df_fe["CityTier"].map({1: 1, 2: 2, 3: 3}).fillna(2).astype(int)

    # Block 6 · Distance & delivery features
    df_fe["Log_WarehouseToHome"] = np.log1p(df_fe["WarehouseToHome"].clip(lower=0))
    df_fe["Far_Warehouse"] = (
        df_fe["WarehouseToHome"] > df_fe["WarehouseToHome"].quantile(0.75)
    ).astype(int)
    df_fe["Distance_x_Dissatisfaction"] = (
        df_fe["WarehouseToHome"] * df_fe["Inv_Satisfaction"]
    )

    # Block 7 · Log transforms
    for col in [
        "CashbackAmount", "OrderAmountHikeFromlastYear", "DaySinceLastOrder",
        "NumberOfAddress", "CouponUsed", "OrderCount"
    ]:
        df_fe[f"Log_{col}"] = np.log1p(df_fe[col].clip(lower=0))

    # Block 8 · Polynomial features
    for col in ["DaySinceLastOrder", "CashbackAmount", "WarehouseToHome"]:
        df_fe[f"{col}_sq"] = df_fe[col] ** 2

    # Block 9 · Cashback tier
    df_fe["Cashback_Tier"] = pd.cut(
        df_fe["CashbackAmount"],
        bins=[-1, 100, 175, 250, 99999],
        labels=["Low", "Medium", "High", "Premium"]
    ).astype(str)

    # Block 10 · Composite churn risk score
    dsl_norm = df_fe["DaySinceLastOrder"] / (df_fe["DaySinceLastOrder"].max() + 1e-9)
    ten_norm = 1 - df_fe["Tenure"] / (df_fe["Tenure"].max() + 1e-9)
    cb_norm = 1 - df_fe["CashbackAmount"] / (df_fe["CashbackAmount"].max() + 1e-9)

    df_fe["Composite_Risk_Score"] = (
        df_fe["Complain"] * 3.0 +
        df_fe["Inv_Satisfaction"] +
        dsl_norm +
        ten_norm +
        cb_norm
    )

    return df_fe

def preprocess_like_notebook(
    df_model_input: pd.DataFrame,
    num_imputer,
    cat_imputer,
    encoder,
    scaler,
    num_cols,
    cat_cols
) -> pd.DataFrame:
    df_proc = df_model_input.copy()

    num_cols_present = [c for c in num_cols if c in df_proc.columns]
    cat_cols_present = [c for c in cat_cols if c in df_proc.columns]

    if num_cols_present:
        df_proc.loc[:, num_cols_present] = num_imputer.transform(df_proc[num_cols_present])

    if cat_cols_present:
        df_proc.loc[:, cat_cols_present] = cat_imputer.transform(df_proc[cat_cols_present])

        encoded = encoder.transform(df_proc[cat_cols_present])
        ohe_cols = encoder.get_feature_names_out(cat_cols_present)

        df_proc_num = df_proc.drop(columns=cat_cols_present).reset_index(drop=True)
        df_proc = pd.concat(
            [
                df_proc_num,
                pd.DataFrame(encoded, columns=ohe_cols, index=df_proc_num.index)
            ],
            axis=1
        )

    numeric_cols_final = df_proc.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols_final:
        df_proc.loc[:, numeric_cols_final] = scaler.transform(df_proc[numeric_cols_final])

    return df_proc

# =========================
# UI header
# =========================
st.title("E-Commerce Churn Prediction Dashboard")

# =========================
# Sidebar controls
# =========================
with st.sidebar:
    st.header("Controls")
    threshold = st.slider("Churn threshold", 0.05, 0.95, 0.50, 0.01)
    uploaded = st.file_uploader("Upload CSV (raw or engineered dataset)", type=["csv"])

    st.divider()
    st.subheader("Debug")
    st.write("Model exists:", MODEL_PATH.exists())
    st.write("Num imputer exists:", NUM_IMPUTER_PATH.exists())
    st.write("Cat imputer exists:", CAT_IMPUTER_PATH.exists())
    st.write("Encoder exists:", ENCODER_PATH.exists())
    st.write("Scaler exists:", SCALER_PATH.exists())
    st.write("Metadata exists:", META_PATH.exists())
    st.write("Optional metrics file:", METRICS_ALL_MODELS_PATH.exists())
    st.write("Optional confusion file:", CONFUSION_ALL_MODELS_PATH.exists())
    st.write("Feature importance (SHAP):", FEATURE_IMPORTANCE_PATH.exists())

# =========================
# Load artifacts
# =========================
@st.cache_resource
def load_artifacts():
    num_imputer = joblib.load(NUM_IMPUTER_PATH)
    cat_imputer = joblib.load(CAT_IMPUTER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)
    model = joblib.load(MODEL_PATH)

    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return num_imputer, cat_imputer, encoder, scaler, model, meta

try:
    num_imputer, cat_imputer, encoder, scaler, model, meta = load_artifacts()
except Exception as e:
    st.error("Failed to load artifacts. Ensure the saved notebook artifacts are beside app.py.")
    st.exception(e)
    st.stop()

target_col = meta.get("target", "Churn")
raw_input_columns = meta.get("raw_input_columns", [])
num_cols = meta.get("num_cols", [])
cat_cols = meta.get("cat_cols", [])

# =========================
# Optional comparison files
# =========================
metrics_all = safe_read_csv(METRICS_ALL_MODELS_PATH)
conf_all = safe_read_csv(CONFUSION_ALL_MODELS_PATH)
feat_imp_external = safe_read_csv(FEATURE_IMPORTANCE_PATH)

# =========================
# No upload -> guidance
# =========================
if uploaded is None:
    st.info("Upload your CSV to score churn. If it includes 'Churn', evaluation will also be computed.")
    st.write("Expected raw/model columns preview:", raw_input_columns[:20])
    st.stop()

# =========================
# Read uploaded dataset
# =========================
try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error("Could not read uploaded CSV.")
    st.exception(e)
    st.stop()

has_target = target_col in df.columns

if has_target:
    y_true = to_binary_series(df[target_col].copy())
else:
    y_true = None

df_original = df.copy()

if has_target:
    df_features = df.drop(columns=[target_col]).copy()
else:
    df_features = df.copy()

required_for_feature_eng = [c for c in RAW_FEATURE_COLUMNS if c in df_features.columns]
if len(required_for_feature_eng) == len(RAW_FEATURE_COLUMNS):
    df_features = add_feature_engineering(df_features)

id_col_detected = pick_customer_id_column(df_features)
if id_col_detected is not None and id_col_detected in df_features.columns:
    df_model_input = df_features.drop(columns=[id_col_detected]).copy()
else:
    df_model_input = df_features.copy()

missing_for_model = [c for c in raw_input_columns if c not in df_model_input.columns]
if len(raw_input_columns) > 0 and missing_for_model:
    st.error("Missing required feature columns for this model:")
    st.write(missing_for_model)
    st.stop()

if raw_input_columns:
    df_model_input = df_model_input[raw_input_columns]

# =========================
# Predict
# =========================
try:
    X_model = preprocess_like_notebook(
        df_model_input=df_model_input,
        num_imputer=num_imputer,
        cat_imputer=cat_imputer,
        encoder=encoder,
        scaler=scaler,
        num_cols=num_cols,
        cat_cols=cat_cols
    )

    proba = model.predict_proba(X_model)[:, 1]
    pred = (proba >= threshold).astype(int)
except Exception as e:
    st.error("Prediction failed.")
    st.exception(e)
    st.stop()

# =========================
# Build output table
# =========================
out = df_original.copy()

engineered_cols_to_add = [c for c in df_features.columns if c not in out.columns]
if engineered_cols_to_add:
    out = pd.concat(
        [out.reset_index(drop=True), df_features[engineered_cols_to_add].reset_index(drop=True)],
        axis=1
    )

out["Churn_probability"] = proba
out["Churn_pred"] = pred
out["Risk_Tier"] = risk_tier_from_proba(proba)

# =========================
# Filters
# =========================
st.subheader("Filters (Demographic & Behavioural)")

filter_cols = [
    "Gender", "MaritalStatus", "PreferredLoginDevice",
    "PreferedOrderCat", "PreferredPaymentMode", "CityTier"
]
available_filters = [c for c in filter_cols if c in out.columns]

if len(available_filters) == 0:
    st.caption("No standard demographic or behavioural filter columns found in this upload.")
    filtered = out.copy()
else:
    cols = st.columns(min(3, len(available_filters)))
    selected = {}

    for i, c in enumerate(available_filters):
        with cols[i % len(cols)]:
            opts = sorted([x for x in out[c].dropna().unique().tolist()])
            sel = st.multiselect(f"{c}", options=opts, default=opts)
            selected[c] = set(sel)

    mask = np.ones(len(out), dtype=bool)
    for c, keep in selected.items():
        mask &= out[c].isin(list(keep))

    filtered = out.loc[mask].copy()

st.caption(f"Rows after filters: {len(filtered):,} / {len(out):,}")

# =========================
# KPI cards
# =========================
k1, k2, k3, k4 = st.columns(4)

if target_col in filtered.columns:
    y_f = to_binary_series(filtered[target_col])
    churn_rate = float((y_f == 1).mean())
    total_churners = int((y_f == 1).sum())
    denom = int(len(y_f))
    k1.metric("Overall Churn Rate (Actual)", f"{churn_rate * 100:.1f}%", f"{total_churners}/{denom}")
else:
    churn_rate = float(filtered["Churn_pred"].mean())
    k1.metric("Overall Churn Rate (Predicted)", f"{churn_rate * 100:.1f}%")

k2.metric("Total Customers at Risk", f"{int(filtered['Churn_pred'].sum()):,}")

if target_col in filtered.columns:
    y_f = to_binary_series(filtered[target_col])
    acc_f = accuracy_score(y_f, filtered["Churn_pred"])
    rec_f = recall_score(y_f, filtered["Churn_pred"], zero_division=0)
    k3.metric("Model Accuracy (XGBoost)", f"{acc_f:.3f}")
    k4.metric("Recall (Churn Catch Rate)", f"{rec_f:.3f}")
else:
    meta_metrics = meta.get("metrics", {})
    k3.metric("Model Accuracy (XGBoost)", f"{meta_metrics.get('accuracy', 0):.3f}")
    k4.metric("Recall (Churn Catch Rate)", f"{meta_metrics.get('recall', 0):.3f}")

st.divider()

# =========================
# Tabs
# =========================
tab_model, tab_drivers, tab_segments, tab_lookup, tab_data = st.tabs([
    "Model Performance & Comparison",
    "Top Influencers (Feature Importance / SHAP)",
    "Customer Risk Segmentation",
    "Individual Customer Lookup",
    "Scored Data"
])

with tab_model:
    st.subheader("XGBoost Performance (filtered segment if labels exist)")

    if target_col in filtered.columns:
        y_f = to_binary_series(filtered[target_col])
        roc = roc_auc_score(y_f, filtered["Churn_probability"]) if len(np.unique(y_f)) == 2 else None
        prec = precision_score(y_f, filtered["Churn_pred"], zero_division=0)
        rec = recall_score(y_f, filtered["Churn_pred"], zero_division=0)
        f1v = f1_score(y_f, filtered["Churn_pred"], zero_division=0)
        acc = accuracy_score(y_f, filtered["Churn_pred"])

        metrics_xgb = pd.DataFrame([{
            "Model": "XGBoost",
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1v,
            "ROC_AUC": roc
        }])
        st.dataframe(metrics_xgb, use_container_width=True)

        cm = confusion_matrix(y_f, filtered["Churn_pred"])
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(cm).plot(ax=ax, colorbar=False)
        ax.set_title("Confusion Matrix (XGBoost, Filtered)")
        st.pyplot(fig)
    else:
        st.info("Upload must include 'Churn' to compute evaluation metrics.")

    st.divider()
    st.subheader("Model Comparison Panel (optional)")

    if metrics_all is None:
        st.warning("metrics_all_models.csv not found. Add it to enable multi-model comparison.")
        st.caption("Expected columns: Model, Accuracy, Precision, Recall, F1, ROC_AUC")
    else:
        plot_bar_metrics(metrics_all)

    if conf_all is not None and "Model" in conf_all.columns:
        st.divider()
        st.subheader("Confusion Matrix Toggle (optional)")
        models = conf_all["Model"].unique().tolist()
        chosen = st.selectbox("Select model", models)
        row = conf_all.loc[conf_all["Model"] == chosen].iloc[0]
        needed = ["TN", "FP", "FN", "TP"]

        if all(k in conf_all.columns for k in needed):
            plot_confusion_from_counts(
                int(row["TN"]), int(row["FP"]), int(row["FN"]), int(row["TP"]),
                title=f"Confusion Matrix ({chosen})"
            )
        else:
            st.warning("confusion_all_models.csv must have columns: Model, TN, FP, FN, TP")

with tab_drivers:
    st.subheader("Top Influencers (Actionable Drivers)")

    guide = pd.DataFrame([
        ["Recency & Activity", "DaySinceLastOrder, Recency_Score, Recency_Bin", "Long gaps since last order signal churn."],
        ["Complaints", "Complain, Complaint_Severity, Chronic_Dissatisfaction", "Dissatisfied customers are more likely to leave."],
        ["Tenure", "Tenure, Lifecycle_Stage, Tenure_sq", "New customers churn more, while long-tenured customers are stickier."],
        ["Order Behaviour", "OrderCount, Order_Frequency, CouponUsed, Coupon_per_Order", "Low engagement increases churn risk."],
        ["Financial", "CashbackAmount, Cashback_per_Order, Cashback_Tier", "Incentives influence retention and repeat purchasing."],
        ["Satisfaction", "SatisfactionScore, Satisfaction_Decay, Inv_Satisfaction", "Direct customer sentiment signal."],
        ["Composite Risk", "Composite_Risk_Score, High_Risk_Flag, Engagement_Score", "Engineered risk features support alerting and prioritisation."],
        ["Logistics", "WarehouseToHome, Far_Warehouse, Distance_x_Dissatisfaction", "Distance can affect delivery experience and churn."]
    ], columns=["Feature Category", "Key Features", "Why It Matters"])
    st.dataframe(guide, use_container_width=True)

    st.divider()
    st.subheader("SHAP Feature Importance")

    if feat_imp_external is not None and set(["feature", "importance"]).issubset(feat_imp_external.columns):
        fi = feat_imp_external.sort_values("importance", ascending=False).head(25)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(fi["feature"].iloc[::-1], fi["importance"].iloc[::-1])
        ax.set_title("Top 25 Features (Mean |SHAP|)")
        ax.set_xlabel("Mean Absolute SHAP Value")
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("feature_importance.csv not found. Export it from the notebook and place it next to app.py.")

with tab_segments:
    st.subheader("Customer Risk Segmentation (Filtered)")

    c1, c2 = st.columns(2)

    with c1:
        plot_pie_counts(filtered["Risk_Tier"], "Risk Tier Distribution (Low / Medium / High / Critical)")

    with c2:
        if "Lifecycle_Stage_Label" in filtered.columns:
            plot_pie_counts(filtered["Lifecycle_Stage_Label"], "Lifecycle Stage Breakdown")
        elif "Lifecycle_Stage" in filtered.columns:
            plot_pie_counts(filtered["Lifecycle_Stage"], "Lifecycle Stage Breakdown")
        else:
            st.info("Lifecycle stage not found in uploaded data.")

    st.divider()
    st.subheader("Cashback Tier vs Churn Rate (Filtered)")

    if "Cashback_Tier" in filtered.columns and target_col in filtered.columns:
        tmp = filtered.copy()
        tmp[target_col] = to_binary_series(tmp[target_col])
        grp = tmp.groupby("Cashback_Tier")[target_col].mean().sort_index()

        fig, ax = plt.subplots()
        ax.bar([str(i) for i in grp.index], grp.values)
        ax.set_title("Churn Rate by Cashback_Tier (Actual)")
        ax.set_xlabel("Cashback_Tier")
        ax.set_ylabel("Churn Rate")
        st.pyplot(fig)
    elif "Cashback_Tier" in filtered.columns:
        st.info("Upload must include 'Churn' to compute churn rate by Cashback_Tier.")
    else:
        st.info("Cashback_Tier not found in uploaded data.")

with tab_lookup:
    st.subheader("Individual Customer Lookup (Filtered)")

    id_col = pick_customer_id_column(filtered)
    if id_col is None:
        st.warning("No CustomerID-like column found. Add a CustomerID column to enable lookup.")
    else:
        st.caption(f"Using ID column: {id_col}")

        ids = sorted(filtered[id_col].astype(str).unique().tolist())
        q = st.text_input("Search CustomerID (type to filter)", value="")
        ids_view = [i for i in ids if q.strip().lower() in i.lower()] if q.strip() else ids

        if len(ids_view) == 0:
            st.info("No matching CustomerID found.")
        else:
            chosen_id = st.selectbox("Select CustomerID", ids_view[:5000])
            row = filtered.loc[filtered[id_col].astype(str) == str(chosen_id)].iloc[0]

            prob = float(row["Churn_probability"])
            tier = str(row["Risk_Tier"])
            pred_label = int(row["Churn_pred"])

            a1, a2, a3 = st.columns(3)
            a1.metric("Churn probability", f"{prob:.3f}")
            a2.metric("Risk tier", tier)
            a3.metric("Predicted churn", str(pred_label))

            st.divider()
            st.subheader("Customer snapshot")

            driver_candidates = [
                "DaySinceLastOrder", "Recency_Score", "Recency_Bin",
                "Complain", "Complaint_Severity",
                "Tenure", "Lifecycle_Stage_Label", "Lifecycle_Stage",
                "OrderCount", "Order_Frequency", "CouponUsed",
                "CashbackAmount", "Cashback_Tier",
                "SatisfactionScore",
                "Composite_Risk_Score", "High_Risk_Flag", "Engagement_Score",
                "WarehouseToHome"
            ]
            snapshot_cols = [id_col, "Risk_Tier", "Churn_probability", "Churn_pred"]
            snapshot_cols += [c for c in driver_candidates if c in row.index]
            snapshot = row[snapshot_cols].to_frame("value")
            st.dataframe(snapshot, use_container_width=True)

            st.divider()
            st.subheader("Recommended action")
            for r in simple_action_recommendations(row):
                st.write("- " + r)

with tab_data:
    st.subheader("Scored Data (Filtered)")
    st.dataframe(filtered.head(200), use_container_width=True)

    st.download_button(
        "Download scored (filtered) CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="scored_filtered.csv",
        mime="text/csv"
    )

    st.divider()
    st.subheader("Probability distribution (Filtered)")
    plot_hist(
        filtered["Churn_probability"].values,
        "Churn Probability Distribution (Filtered)",
        "Probability"
    )

    if target_col in filtered.columns:
        st.caption("Actual churn rate under current filters:")
        st.write(f"{to_binary_series(filtered[target_col]).mean() * 100:.2f}%")