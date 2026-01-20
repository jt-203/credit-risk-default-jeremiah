"""
app.py

Streamlit dashboard for exploring credit risk predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "credit_risk_model.joblib"
META_PATH = ARTIFACT_DIR / "model_metadata.joblib"


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    metadata = joblib.load(META_PATH)
    return model, metadata


def main():
    st.title("Credit Risk Default Model — Jeremiah Tshinyama")
    st.write(
        "This dashboard uses a machine learning model trained on historical loan data "
        "to estimate the probability of default at origination."
    )

    if not MODEL_PATH.exists() or not META_PATH.exists():
        st.error(
            "Model artifacts not found. Please run `train_models.py` first to train and save the model."
        )
        return

    model, metadata = load_model()
    columns = metadata["columns"]
    auc = metadata.get("auc", None)

    if auc is not None:
        st.metric("Validation ROC-AUC", f"{auc:.3f}")

    st.subheader("Input Borrower & Loan Characteristics")

    inputs = {}
    for col in columns:
        # Simple heuristic: numeric vs categorical
        if col.lower().endswith(("amt", "inc", "dti", "int_rate", "installment")) or col in ["loan_amnt", "annual_inc"]:
            val = st.number_input(col, value=0.0)
            inputs[col] = val
        else:
            txt = st.text_input(col, value="")
            inputs[col] = txt

    if st.button("Predict Default Probability"):
        X = pd.DataFrame([inputs], columns=columns)
        proba = model.predict_proba(X)[:, 1][0]
        st.metric("Estimated Probability of Default", f"{proba:.2%}")

        st.write(
            "You can extend this app by adding feature importance plots, "
            "policy scenarios, or segment-level analytics."
        )


if __name__ == "__main__":
    main()
