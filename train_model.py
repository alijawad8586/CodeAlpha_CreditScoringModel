"""Train and evaluate credit-risk classifiers on OpenML German Credit data."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
IMAGES = ROOT / "images"


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add business-friendly features used by training and inference."""
    data = frame.copy()
    if {"credit_amount", "duration"} <= set(data.columns):
        data["credit_amount_per_month"] = (
            pd.to_numeric(data["credit_amount"]) /
            pd.to_numeric(data["duration"]).replace(0, 1)
        )
    if "age" in data:
        data["age_group"] = pd.cut(
            pd.to_numeric(data["age"]),
            bins=[0, 25, 35, 50, 65, np.inf],
            labels=["young", "young_adult", "middle_age", "senior", "elderly"],
        )
    if {"credit_amount", "existing_credits"} <= set(data.columns):
        data["credit_per_existing_account"] = (
            pd.to_numeric(data["credit_amount"]) /
            pd.to_numeric(data["existing_credits"]).replace(0, 1)
        )
    return data


def load_data() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    dataset = fetch_openml(data_id=31, as_frame=True, parser="auto")
    raw = dataset.data.copy()
    target = (
        dataset.target.astype(str).str.lower().map({"good": 0, "bad": 1})
    )
    if target.isna().any():
        raise ValueError("Unexpected target labels in OpenML dataset 31.")
    combined = raw.copy()
    combined["target"] = target.to_numpy()
    combined = combined.drop_duplicates().reset_index(drop=True)
    raw_columns = raw.columns.tolist()
    return engineer_features(combined.drop(columns="target")), combined["target"], raw_columns


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numerical = X.select_dtypes(include=np.number).columns.tolist()
    categorical = X.columns.difference(numerical).tolist()
    return ColumnTransformer(
        [
            (
                "numerical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numerical,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )


def model_candidates() -> dict[str, object]:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=10,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }


def make_schema(raw_X: pd.DataFrame) -> dict:
    fields = []
    for column in raw_X.columns:
        series = raw_X[column]
        if pd.api.types.is_numeric_dtype(series):
            fields.append(
                {
                    "name": column,
                    "kind": "number",
                    "default": float(series.median()),
                    "min": float(series.min()),
                    "max": float(series.max()),
                }
            )
        else:
            values = sorted(series.dropna().astype(str).unique().tolist())
            fields.append(
                {
                    "name": column,
                    "kind": "category",
                    "default": str(series.mode().iloc[0]),
                    "options": values,
                }
            )
    return {"fields": fields}


def train_and_save() -> pd.DataFrame:
    X, y, raw_columns = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    preprocessor = build_preprocessor(X_train)
    results, trained = [], {}
    IMAGES.mkdir(exist_ok=True)

    for name, estimator in model_candidates().items():
        pipeline = Pipeline(
            [("preprocessor", clone(preprocessor)), ("classifier", estimator)]
        )
        pipeline.fit(X_train, y_train)
        prediction = pipeline.predict(X_test)
        probability = pipeline.predict_proba(X_test)[:, 1]
        results.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, prediction),
                "Precision": precision_score(y_test, prediction, zero_division=0),
                "Recall": recall_score(y_test, prediction, zero_division=0),
                "F1-Score": f1_score(y_test, prediction, zero_division=0),
                "ROC-AUC": roc_auc_score(y_test, probability),
            }
        )
        trained[name] = pipeline

    results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False)
    best_name = str(results_df.iloc[0]["Model"])
    best_model = trained[best_name]
    joblib.dump(best_model, ROOT / "credit_scoring_model.pkl")
    results_df.to_csv(ROOT / "model_results.csv", index=False)

    raw_X = X[raw_columns]
    schema = make_schema(raw_X)
    schema.update(
        {
            "best_model": best_name,
            "best_roc_auc": float(results_df.iloc[0]["ROC-AUC"]),
            "raw_columns": raw_columns,
        }
    )
    (ROOT / "model_schema.json").write_text(
        json.dumps(schema, indent=2), encoding="utf-8"
    )

    ConfusionMatrixDisplay.from_estimator(
        best_model,
        X_test,
        y_test,
        display_labels=["Good Credit", "Bad Credit"],
        cmap="Blues",
    )
    plt.title(f"Confusion Matrix — {best_name}")
    plt.tight_layout()
    plt.savefig(IMAGES / "confusion_matrix.png", dpi=160)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, pipeline in trained.items():
        RocCurveDisplay.from_estimator(pipeline, X_test, y_test, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "--", label="Random Guess")
    ax.set_title("ROC Curves — Credit Scoring Models")
    ax.legend()
    fig.tight_layout()
    fig.savefig(IMAGES / "roc_curve.png", dpi=160)
    plt.close(fig)

    print(f"Dataset: {len(X)} rows, {len(raw_columns)} original features")
    print(f"Train/test: {len(X_train)}/{len(X_test)}")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nBest model: {best_name}")
    return results_df


if __name__ == "__main__":
    train_and_save()
