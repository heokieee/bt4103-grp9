
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

TARGET_COL = "Churn"
SOURCE_VALUE = "client_submission"
SUBMISSIONS_COLLECTION = "customer_submissions"
CURRENT_COLLECTION = "current_customers"

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

def get_next_customer_id(db, firestore_module, current_collection: str = CURRENT_COLLECTION) -> int:
    docs = (
        db.collection(current_collection)
        .order_by("CustomerID", direction=firestore_module.Query.DESCENDING)
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

def validate_submission(payload: dict[str, Any], schema: dict[str, Any], target_col: str = TARGET_COL) -> list[str]:
    errors: list[str] = []

    for col in schema["columns"]:
        value = payload.get(col)

        if col == target_col and value is None:
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

def write_to_firestore(
    db,
    firestore_module,
    payload: dict[str, Any],
    errors: list[str],
    source_value: str = SOURCE_VALUE,
    submissions_collection: str = SUBMISSIONS_COLLECTION,
    current_collection: str = CURRENT_COLLECTION,
) -> tuple[str, str | None]:
    now = firestore_module.SERVER_TIMESTAMP

    submission_doc = {
        **payload,
        "source": source_value,
        "submitted_at": now,
        "validation_status": "valid" if not errors else "invalid",
        "validation_errors": errors,
    }

    created = db.collection(submissions_collection).add(submission_doc)
    submission_id = created[1].id

    promoted_id: str | None = None
    if not errors:
        promoted_doc = {
            **payload,
            "source": source_value,
            "promoted_at": now,
            "submission_id": submission_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        promoted_ref = db.collection(current_collection).document(submission_id)
        promoted_ref.set(promoted_doc)
        promoted_id = submission_id

    return submission_id, promoted_id
