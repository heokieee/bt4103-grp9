from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class FakeFirestoreModule:
    SERVER_TIMESTAMP = "SERVER_TIMESTAMP"

    class Query:
        DESCENDING = "DESCENDING"


@dataclass
class FakeDoc:
    id: str
    data: dict

    def to_dict(self):
        return self.data


class FakeDocumentRef:
    def __init__(self, collection, doc_id):
        self.collection = collection
        self.id = doc_id

    def set(self, payload):
        self.collection.documents[self.id] = payload


class FakeCollection:
    def __init__(self, name, documents=None):
        self.name = name
        self.documents = documents or {}
        self._ordered = False
        self._limit = None
        self._order_field = None
        self._direction = None

    def add(self, payload):
        new_id = f"{self.name}_{len(self.documents) + 1}"
        self.documents[new_id] = payload
        return None, FakeDoc(new_id, payload)

    def document(self, doc_id):
        return FakeDocumentRef(self, doc_id)

    def order_by(self, field, direction=None):
        self._ordered = True
        self._order_field = field
        self._direction = direction
        return self

    def limit(self, n):
        self._limit = n
        return self

    def stream(self):
        docs = [FakeDoc(doc_id, data) for doc_id, data in self.documents.items()]
        if self._ordered and self._order_field is not None:
            docs.sort(
                key=lambda d: d.to_dict().get(self._order_field, 0),
                reverse=(self._direction == FakeFirestoreModule.Query.DESCENDING),
            )
        if self._limit is not None:
            docs = docs[: self._limit]
        return docs


class FakeDB:
    def __init__(self):
        self.collections = {}

    def collection(self, name):
        if name not in self.collections:
            self.collections[name] = FakeCollection(name)
        return self.collections[name]


class FakePipelineArray:
    def transform(self, df):
        n = len(df)
        return np.array([[10.0, 20.0, 30.0] for _ in range(n)])


class FakePipelineDataFrame:
    def transform(self, df):
        n = len(df)
        return pd.DataFrame({"p1": [1.0] * n, "p2": [2.0] * n})


class FakeModel:
    def __init__(self, positive_probs):
        self.positive_probs = np.array(positive_probs, dtype=float)

    def predict_proba(self, X):
        n = len(X)
        p1 = self.positive_probs
        if len(p1) == 1:
            p1 = np.repeat(p1, n)
        p0 = 1 - p1
        return np.column_stack([p0, p1])


@pytest.fixture
def fake_firestore():
    return FakeFirestoreModule


@pytest.fixture
def fake_db():
    db = FakeDB()
    current = db.collection("current_customers")
    current.documents = {
        "doc_1": {"CustomerID": 1001},
        "doc_2": {"CustomerID": 1005},
    }
    return db


@pytest.fixture
def sample_schema():
    return {
        "columns": [
            "CustomerID",
            "Tenure",
            "PreferredLoginDevice",
            "CityTier",
            "Complain",
            "Churn",
        ],
        "numeric_cols": ["CustomerID", "Tenure", "CityTier", "Complain", "Churn"],
        "categorical_cols": ["PreferredLoginDevice"],
        "numeric_stats": {
            "CustomerID": {"min": 1.0, "max": 999999.0, "median": 1000.0, "is_int": True},
            "Tenure": {"min": 0.0, "max": 61.0, "median": 8.0, "is_int": True},
            "CityTier": {"min": 1.0, "max": 3.0, "median": 1.0, "is_int": True},
            "Complain": {"min": 0.0, "max": 1.0, "median": 0.0, "is_int": True},
            "Churn": {"min": 0.0, "max": 1.0, "median": 0.0, "is_int": True},
        },
        "categorical_values": {
            "PreferredLoginDevice": ["Computer", "Mobile Phone"],
        },
    }


@pytest.fixture
def valid_payload():
    return {
        "CustomerID": 1006,
        "Tenure": 8,
        "PreferredLoginDevice": "Mobile Phone",
        "CityTier": 2,
        "Complain": 0,
        "Churn": None,
    }


@pytest.fixture
def sample_raw_df():
    return pd.DataFrame(
        {
            "CustomerID": [1006, 1007],
            "Tenure": [8, 14],
            "PreferredLoginDevice": ["Mobile Phone", "Computer"],
            "CityTier": [2, 1],
            "Complain": [0, 1],
            "Churn": [0, 1],
            "source": ["client_submission", "client_submission"],
            "submission_id": ["s1", "s2"],
            "submitted_at": ["t1", "t2"],
        }
    )


@pytest.fixture
def project_root():
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def real_dataset_csv(project_root):
    return project_root / "E Commerce Dataset.csv"


@pytest.fixture
def metadata_dict(project_root):
    with open(project_root / "ensemble_metadata.json", "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def mock_preprocessor_array():
    return FakePipelineArray()


@pytest.fixture
def mock_preprocessor_df():
    return FakePipelineDataFrame()


@pytest.fixture
def mock_rf_model():
    return FakeModel([0.2])


@pytest.fixture
def mock_xgb_model():
    return FakeModel([0.6])


@pytest.fixture
def mock_lgbm_model():
    return FakeModel([0.8])