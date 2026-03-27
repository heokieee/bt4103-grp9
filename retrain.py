from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from firebase_admin import credentials, firestore, initialize_app
from google.cloud import storage
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from xgboost import XGBClassifier


BASE = Path(__file__).resolve().parent

CSV_PATH = BASE / "E Commerce Dataset.csv"
PIPELINE_PATH = BASE / "ensemble_preprocessing_pipeline.joblib"
RF_MODEL_PATH = BASE / "ensemble_rf_model.joblib"
XGB_MODEL_PATH = BASE / "ensemble_xgb_model.joblib"
LGBM_MODEL_PATH = BASE / "ensemble_lgbm_model.joblib"
META_PATH = BASE / "ensemble_metadata.json"
RETRAIN_LOG_PATH = BASE / "retrain_log.csv"

COLLECTION_NAME = "current_customers"
TARGET_COL = "Churn"
ID_COL = "CustomerID"
MIN_FIRESTORE_ROWS = 200
TEST_SIZE = 0.2
RANDOM_STATE = 42
K_FEATURES = 50

ALL_COLUMNS = [
    "CustomerID",
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
    "Churn",
    "PreferredLoginDevice",
    "PreferredPaymentMode",
    "Gender",
    "PreferedOrderCat",
    "MaritalStatus",
]

FEATURE_COLUMNS = [c for c in ALL_COLUMNS if c != TARGET_COL]

NUMERIC_COLUMNS = [
    "CustomerID",
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
]

CATEGORICAL_COLUMNS = [
    "PreferredLoginDevice",
    "PreferredPaymentMode",
    "Gender",
    "PreferedOrderCat",
    "MaritalStatus",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain ensemble churn models from Firestore data.")
    parser.add_argument(
        "--service-account",
        required=True,
        help="Path to Firestore service account JSON file.",
    )
    parser.add_argument(
        "--bucket",
        required=False,
        default="",
        help="Optional GCS bucket name to upload refreshed artifacts.",
    )
    parser.add_argument(
        "--bucket-prefix",
        required=False,
        default="ensemble",
        help="Optional GCS object prefix for model artifacts.",
    )
    return parser.parse_args()


def build_blob_name(prefix: str, file_name: str) -> str:
    clean_prefix = prefix.strip().strip("/")
    if not clean_prefix:
        return file_name
    return f"{clean_prefix}/{file_name}"


def upload_artifacts_to_bucket(
    service_account_path: str,
    bucket_name: str,
    bucket_prefix: str,
) -> list[str]:
    print(f"Connecting to GCS bucket: {bucket_name}")

    try:
        client = storage.Client.from_service_account_json(service_account_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to create GCS client: {exc}") from exc

    try:
        bucket = client.bucket(bucket_name)
        # Force metadata fetch to confirm the bucket exists and is accessible.
        bucket.reload()
    except Exception as exc:
        raise RuntimeError(
            f"Cannot access bucket '{bucket_name}'. "
            "Check the bucket name and that your service account has "
            "'Storage Object Admin' role in GCP IAM.\n"
            f"Original error: {exc}"
        ) from exc

    files_to_upload = [
        PIPELINE_PATH,
        RF_MODEL_PATH,
        XGB_MODEL_PATH,
        LGBM_MODEL_PATH,
        META_PATH,
        RETRAIN_LOG_PATH,
    ]

    uploaded_objects: list[str] = []
    for file_path in files_to_upload:
        if not file_path.exists():
            print(f"  WARNING: {file_path.name} not found locally, skipping.")
            continue
        blob_name = build_blob_name(bucket_prefix, file_path.name)
        blob = bucket.blob(blob_name)
        print(f"  Uploading {file_path.name} -> gs://{bucket_name}/{blob_name}")
        blob.upload_from_filename(str(file_path))
        uploaded_objects.append(blob_name)
        print("  Done.")

    return uploaded_objects


def normalize_binary(series: pd.Series) -> pd.Series:
    mapping = {
        "Yes": 1,
        "No": 0,
        "yes": 1,
        "no": 0,
        "1": 1,
        "0": 0,
        1: 1,
        0: 0,
        True: 1,
        False: 0,
    }
    mapped = series.map(mapping)
    merged = mapped.where(mapped.notna(), series)
    return pd.to_numeric(merged, errors="coerce")


def coerce_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in ALL_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA

    out = out[ALL_COLUMNS]

    for col in NUMERIC_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in CATEGORICAL_COLUMNS:
        out[col] = out[col].astype("string")

    out[TARGET_COL] = normalize_binary(out[TARGET_COL])
    out = out.dropna(subset=[TARGET_COL]).copy()
    out[TARGET_COL] = out[TARGET_COL].astype(int)

    return out


def fetch_firestore_dataframe(service_account_path: str) -> pd.DataFrame:
    import firebase_admin

    cred = credentials.Certificate(service_account_path)
    # Guard against double-initialisation (e.g. during testing or reimport).
    if not firebase_admin._apps:
        app = initialize_app(cred)
    else:
        app = firebase_admin.get_app()
    db = firestore.client(app=app)

    rows: list[dict[str, Any]] = []
    for doc in db.collection(COLLECTION_NAME).stream():
        record = doc.to_dict() or {}
        rows.append(record)

    if not rows:
        return pd.DataFrame(columns=ALL_COLUMNS)

    return pd.DataFrame(rows)


def choose_training_source(firestore_df: pd.DataFrame) -> tuple[pd.DataFrame, str, int]:
    firestore_count = len(firestore_df)

    if firestore_count >= MIN_FIRESTORE_ROWS:
        return firestore_df, "firestore", firestore_count

    csv_df = pd.read_csv(CSV_PATH)
    return csv_df, "csv_fallback", firestore_count


def build_preprocessing_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[Pipeline, list[str]]:
    numeric_cols = [c for c in X_train.columns if c in NUMERIC_COLUMNS and c != ID_COL]
    categorical_cols = [c for c in X_train.columns if c in CATEGORICAL_COLUMNS]

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "poly",
                PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
            ),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )
    preprocessor.set_output(transform="pandas")

    selector = SelectKBest(score_func=f_classif, k=K_FEATURES)

    full_pipeline = Pipeline(
        [
            ("preprocessing", preprocessor),
            ("feature_selection", selector),
        ]
    )

    full_pipeline.fit(X_train, y_train)

    feature_names = full_pipeline.named_steps["preprocessing"].get_feature_names_out()
    selected_mask = full_pipeline.named_steps["feature_selection"].get_support()
    selected_features = feature_names[selected_mask].tolist()

    return full_pipeline, selected_features


def transform_with_selected_features(
    pipeline: Pipeline,
    X: pd.DataFrame,
    selected_features: list[str],
) -> pd.DataFrame:
    transformed = pipeline.transform(X)
    if isinstance(transformed, pd.DataFrame):
        return transformed
    return pd.DataFrame(transformed, columns=selected_features)


def compute_recall_weights(
    y_test: pd.Series,
    rf_model,
    xgb_model,
    lgbm_model,
    X_test_processed: pd.DataFrame,
) -> dict[str, float]:
    recalls = {
        "w_rf": float(recall_score(y_test, rf_model.predict(X_test_processed))),
        "w_xgb": float(recall_score(y_test, xgb_model.predict(X_test_processed))),
        "w_lgbm": float(recall_score(y_test, lgbm_model.predict(X_test_processed))),
    }
    total = recalls["w_rf"] + recalls["w_xgb"] + recalls["w_lgbm"]
    if total == 0:
        return {"w_rf": 1 / 3, "w_xgb": 1 / 3, "w_lgbm": 1 / 3}
    return {k: v / total for k, v in recalls.items()}


def predict_ensemble(
    X_processed: pd.DataFrame,
    rf_model,
    xgb_model,
    lgbm_model,
    weights: dict[str, float],
) -> pd.Series:
    rf_proba = rf_model.predict_proba(X_processed)[:, 1]
    xgb_proba = xgb_model.predict_proba(X_processed)[:, 1]
    lgbm_proba = lgbm_model.predict_proba(X_processed)[:, 1]

    y_proba = (
        weights["w_rf"] * rf_proba
        + weights["w_xgb"] * xgb_proba
        + weights["w_lgbm"] * lgbm_proba
    )

    return pd.Series((y_proba >= 0.5).astype(int))


def evaluate_models(
    y_true: pd.Series,
    pipeline: Pipeline,
    selected_features: list[str],
    rf_model,
    xgb_model,
    lgbm_model,
    weights: dict[str, float],
    X_test_raw: pd.DataFrame,
) -> dict[str, float]:
    X_test_processed = transform_with_selected_features(pipeline, X_test_raw, selected_features)
    y_pred = predict_ensemble(X_test_processed, rf_model, xgb_model, lgbm_model, weights)
    return {
        "f1": float(f1_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
    }


def append_retrain_log(row: dict[str, Any]) -> None:
    log_df = pd.DataFrame([row])
    if RETRAIN_LOG_PATH.exists():
        existing = pd.read_csv(RETRAIN_LOG_PATH)
        combined = pd.concat([existing, log_df], ignore_index=True)
    else:
        combined = log_df
    combined.to_csv(RETRAIN_LOG_PATH, index=False)


def main() -> None:
    args = parse_args()

    firestore_df = fetch_firestore_dataframe(args.service_account)
    source_df, data_source, firestore_count = choose_training_source(firestore_df)

    prepared_df = coerce_schema(source_df)
    if prepared_df.empty:
        raise ValueError("No labeled rows found after schema coercion. Cannot retrain.")

    if prepared_df[TARGET_COL].nunique() < 2:
        raise ValueError("Training data must contain both churn classes.")

    X = prepared_df[FEATURE_COLUMNS].drop(columns=[ID_COL], errors="ignore")
    y = prepared_df[TARGET_COL]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        stratify=y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    # Load existing artifacts for before-metrics comparison.
    # On a cold start (no prior joblib files), skip gracefully.
    prior_artifacts_exist = all(
        p.exists()
        for p in [PIPELINE_PATH, RF_MODEL_PATH, XGB_MODEL_PATH, LGBM_MODEL_PATH, META_PATH]
    )

    before_metrics: dict[str, float] = {"f1": float("nan"), "recall": float("nan")}

    if prior_artifacts_exist:
        old_pipeline = joblib.load(PIPELINE_PATH)
        old_rf_model = joblib.load(RF_MODEL_PATH)
        old_xgb_model = joblib.load(XGB_MODEL_PATH)
        old_lgbm_model = joblib.load(LGBM_MODEL_PATH)
        with open(META_PATH, "r", encoding="utf-8") as f:
            old_meta = json.load(f)

        old_selected_features = old_meta.get("selected_features", [])
        old_weights = old_meta.get("weights", {"w_rf": 1 / 3, "w_xgb": 1 / 3, "w_lgbm": 1 / 3})

        try:
            before_metrics = evaluate_models(
                y_true=y_test,
                pipeline=old_pipeline,
                selected_features=old_selected_features,
                rf_model=old_rf_model,
                xgb_model=old_xgb_model,
                lgbm_model=old_lgbm_model,
                weights=old_weights,
                X_test_raw=X_test_raw,
            )
        except Exception as exc:
            print(f"  WARNING: Could not compute before-metrics (pipeline mismatch?): {exc}")
            before_metrics = {"f1": float("nan"), "recall": float("nan")}
    else:
        print("  No prior model artifacts found — cold start. Skipping before-metrics.")

    new_pipeline, selected_features = build_preprocessing_pipeline(X_train_raw, y_train)
    X_train_processed = transform_with_selected_features(new_pipeline, X_train_raw, selected_features)
    X_test_processed = transform_with_selected_features(new_pipeline, X_test_raw, selected_features)

    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_processed, y_train)

    rf_model = clone(old_rf_model)
    xgb_model = clone(old_xgb_model)
    lgbm_model = clone(old_lgbm_model)

    # Keep explicit model types for readability in logs/errors.
    if not isinstance(xgb_model, XGBClassifier):
        raise TypeError("Loaded XGBoost artifact is not an XGBClassifier.")
    if not isinstance(lgbm_model, LGBMClassifier):
        raise TypeError("Loaded LightGBM artifact is not an LGBMClassifier.")

    rf_model.fit(X_train_smote, y_train_smote)
    xgb_model.fit(X_train_smote, y_train_smote)
    lgbm_model.fit(X_train_smote, y_train_smote)

    new_weights = compute_recall_weights(y_test, rf_model, xgb_model, lgbm_model, X_test_processed)

    after_metrics = evaluate_models(
        y_true=y_test,
        pipeline=new_pipeline,
        selected_features=selected_features,
        rf_model=rf_model,
        xgb_model=xgb_model,
        lgbm_model=lgbm_model,
        weights=new_weights,
        X_test_raw=X_test_raw,
    )

    retrained_at = datetime.now(timezone.utc).isoformat()

    metadata = {
        "target": TARGET_COL,
        "id_cols": [ID_COL],
        "raw_input_columns": FEATURE_COLUMNS,
        "selected_features": selected_features,
        "weights": {k: float(v) for k, v in new_weights.items()},
        "retrained_at": retrained_at,
    }

    joblib.dump(new_pipeline, PIPELINE_PATH)
    joblib.dump(rf_model, RF_MODEL_PATH)
    joblib.dump(xgb_model, XGB_MODEL_PATH)
    joblib.dump(lgbm_model, LGBM_MODEL_PATH)

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    append_retrain_log(
        {
            "retrained_at": retrained_at,
            "data_source": data_source,
            "firestore_record_count": int(firestore_count),
            "training_row_count": int(len(prepared_df)),
            "before_f1": before_metrics["f1"],
            "before_recall": before_metrics["recall"],
            "after_f1": after_metrics["f1"],
            "after_recall": after_metrics["recall"],
        }
    )

    uploaded_objects: list[str] = []
    if not args.bucket.strip():
        print("\nNo --bucket argument provided. Skipping GCS upload.")
        print(
            "To upload, run with: --bucket bt4013team9-664f6.firebasestorage.app "
            "--bucket-prefix ensemble"
        )
    else:
        bucket_root = f"gs://{args.bucket.strip()}"
        if args.bucket_prefix.strip().strip("/"):
            bucket_root = f"{bucket_root}/{args.bucket_prefix.strip().strip('/')}"
        print(f"\nUploading artifacts to {bucket_root} ...")
        try:
            uploaded_objects = upload_artifacts_to_bucket(
                service_account_path=args.service_account,
                bucket_name=args.bucket.strip(),
                bucket_prefix=args.bucket_prefix,
            )
            print(f"Upload complete. {len(uploaded_objects)} file(s) written:")
            for blob_name in uploaded_objects:
                print(f"  gs://{args.bucket.strip()}/{blob_name}")
        except RuntimeError as exc:
            print(f"\nERROR during GCS upload:\n{exc}")
            print("Local joblib files have been saved. Fix the bucket issue and re-upload manually.")

    # ── Final summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RETRAINING COMPLETE")
    print("=" * 60)
    print(f"  Data source       : {data_source}")
    print(f"  Firestore records : {firestore_count}")
    print(f"  Rows used         : {len(prepared_df)}")
    before_f1 = before_metrics['f1']
    before_rec = before_metrics['recall']
    after_f1 = after_metrics['f1']
    after_rec = after_metrics['recall']
    print(f"  Before  F1/Recall : {before_f1:.4f} / {before_rec:.4f}" if not (
        before_f1 != before_f1) else "  Before  F1/Recall : N/A (cold start)")
    print(f"  After   F1/Recall : {after_f1:.4f} / {after_rec:.4f}")
    print(f"  GCS artifacts     : {len(uploaded_objects)} uploaded")


if __name__ == "__main__":
    main()