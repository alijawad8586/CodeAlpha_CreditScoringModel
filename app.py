"""Streamlit decision-support interface for the trained credit scoring model."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from train_model import engineer_features

ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="Credit Risk Scoring", page_icon="🏦", layout="wide")


@st.cache_resource
def load_artifacts():
    model = joblib.load(ROOT / "credit_scoring_model.pkl")
    schema = json.loads((ROOT / "model_schema.json").read_text(encoding="utf-8"))
    return model, schema


st.title("🏦 Credit Risk Scoring")
st.caption("Decision-support demo — this prediction is not a final lending decision.")

try:
    model, schema = load_artifacts()
except FileNotFoundError:
    st.error("Model files are missing. Run `python train_model.py` first.")
    st.stop()

with st.sidebar:
    st.header("Model information")
    st.metric("Selected model", schema["best_model"])
    st.metric("Test ROC-AUC", f'{schema["best_roc_auc"]:.3f}')

st.subheader("Applicant financial profile")
values = {}
columns = st.columns(2)
for index, field in enumerate(schema["fields"]):
    label = field["name"].replace("_", " ").title()
    with columns[index % 2]:
        if field["kind"] == "number":
            values[field["name"]] = st.number_input(
                label,
                min_value=field["min"],
                max_value=field["max"],
                value=field["default"],
            )
        else:
            default_index = field["options"].index(field["default"])
            values[field["name"]] = st.selectbox(
                label, field["options"], index=default_index
            )

if st.button("Assess credit risk", type="primary", use_container_width=True):
    applicant = engineer_features(pd.DataFrame([values]))
    prediction = int(model.predict(applicant)[0])
    probability = float(model.predict_proba(applicant)[0, 1])
    st.subheader("Assessment")
    left, right = st.columns(2)
    left.metric("Risk probability", f"{probability:.1%}")
    right.metric(
        "Model decision",
        "High Risk" if prediction == 1 else "Low Risk",
    )
    if prediction == 1:
        st.warning("The model flags this profile for additional human review.")
    else:
        st.success("The model estimates lower credit risk for this profile.")
    st.info(
        "Use this result with affordability checks, identity verification, "
        "policy rules, and a qualified human review."
    )
