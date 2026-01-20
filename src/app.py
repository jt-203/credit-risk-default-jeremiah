# src/app.py

import joblib
import numpy as np
import pandas as pd
import streamlit as st


@st.cache_resource
def load_model():
    data = joblib.load("artifacts/credit_default_model.joblib")
    return data["model"], data["numeric_features"], data["categorical_features"]


def get_policy_thresholds(tolerance: str):
    """
    Returns (approve, review) thresholds for default probability.
    Above 'review' -> reject.
    """
    if tolerance == "Conservative":
        return 0.05, 0.15
    elif tolerance == "Balanced":
        return 0.08, 0.22
    else:  # Aggressive
        return 0.12, 0.30


def main():
    st.set_page_config(
        page_title="Credit Risk Default Model – Jeremiah Tshinyama",
        layout="centered",
    )

    st.title("Credit Risk Default Model")
    st.caption("Built by **Jeremiah Tshinyama**")

    st.markdown(
        """
This app estimates the probability that a personal loan will default based on
borrower and loan characteristics.  
Use the controls in the sidebar to adjust inputs and risk tolerance.
        """
    )

    # Load model
    try:
        model, numeric_features, categorical_features = load_model()
    except Exception as e:
        st.error(
            "Model not found. Please train the model first by running "
            "`python src/train_models.py --data_path data/your_loans.csv` "
            "from the project root."
        )
        st.exception(e)
        return

    # Sidebar: risk tolerance + inputs
    st.sidebar.header("Lender Settings")

    tolerance = st.sidebar.selectbox(
        "Risk tolerance profile",
        ["Conservative", "Balanced", "Aggressive"],
        index=1,
        help="Conservative = fewer approvals, lower loss risk. Aggressive = more approvals, higher loss risk.",
    )

    approve_thr, review_thr = get_policy_thresholds(tolerance)

    st.sidebar.markdown("### Borrower & Loan Inputs")

    # Default input values (generic, adjust as needed)
    loan_amnt = st.sidebar.number_input("Loan amount ($)", min_value=500.0, max_value=80000.0, value=15000.0, step=500.0)
    int_rate = st.sidebar.number_input("Interest rate (%)", min_value=5.0, max_value=35.0, value=14.0, step=0.5)
    annual_inc = st.sidebar.number_input("Annual income ($)", min_value=10000.0, max_value=300000.0, value=65000.0, step=5000.0)
    dti = st.sidebar.number_input("Debt-to-income ratio (%)", min_value=0.0, max_value=60.0, value=18.0, step=1.0)
    revol_util = st.sidebar.number_input("Revolving utilization (%)", min_value=0.0, max_value=150.0, value=45.0, step=1.0)
    open_acc = st.sidebar.number_input("Open credit lines", min_value=0, max_value

