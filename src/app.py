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
    st.caption("Developed by **Jeremiah Tshinyama**")

    st.markdown(
        """
This tool estimates the probability that a personal loan may default based on
borrower features and loan characteristics.

👉 Adjust values in the sidebar to simulate borrower profiles  
👉 Choose a lender risk tolerance to get suggested lending decisions
        """
    )

    # Load saved model
    try:
        model, numeric_features, categorical_features = load_model()
    except Exception as e:
        st.error(
            "Model artifact not found. Please train a model first by running:\n"
            "`python src/train_models.py --data_path data/your_loans.csv`"
        )
        st.exception(e)
        return

    # Sidebar inputs
    st.sidebar.header("Borrower & Loan Inputs")
    tolerance = st.sidebar.selectbox(
        "Lender Risk Tolerance",
        ["Conservative", "Balanced", "Aggressive"],
        help="Conservative lenders approve fewer borrowers with lower default risk.",
    )

    approve_thr, review_thr = get_policy_thresholds(tolerance)

    loan_amnt = st.sidebar.number_input("Loan Amount ($)", 500.0, 80000.0, 12000.0)
    int_rate = st.sidebar.number_input("Interest Rate (%)", 5.0, 35.0, 14.5)
    annual_inc = st.sidebar.number_input("Annual Income ($)", 10000.0, 300000.0, 55000.0)
    dti = st.sidebar.number_input("Debt-to-Income (%)", 0.0, 60.0, 18.0)
    revol_util = st.sidebar.number_input("Revolving Utilization (%)", 0.0, 150.0, 45.0)
    open_acc = st.sidebar.number_input("Open Credit Lines", 0, 50, 8)
    total_acc = st.sidebar.number_input("Total Credit Accounts", 1, 100, 24)

    term = st.sidebar.selectbox("Term", ["36 months", "60 months"])
    grade = st.sidebar.selectbox("Credit Grade", list("ABCDEFG"))
    home_ownership = st.sidebar.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
    purpose = st.sidebar.selectbox("Loan Purpose", ["debt_consolidation", "credit_card", "home_improvement", "other"])
    verification_status = st.sidebar.selectbox("Verification Status", ["Verified", "Source Verified", "Not Verified"])

    if st.sidebar.button("Predict Default Risk"):
        input_df = pd.DataFrame(
            [{
                "loan_amnt": loan_amnt,
                "int_rate": int_rate,
                "annual_inc": annual_inc,
                "dti": dti,
                "revol_util": revol_util,
                "open_acc": open_acc,
                "total_acc": total_acc,
                "term": term,
                "grade": grade,
                "home_ownership": home_ownership,
                "purpose": purpose,
                "verification_status": verification_status,
            }]
        )

        pred_proba = model.predict_proba(input_df)[0][1]
        st.subheader(f"📊 Estimated Default Probability: **{pred_proba:.1%}**")

        # Lending policy decision
        if pred_proba < approve_thr:
            st.success("✔ Recommendation: **APPROVE** under current risk tolerance.")
        elif pred_proba < review_thr:
            st.warning("⚠ Recommendation: **REVIEW / ADD CONDITIONS** (higher rate, collateral, etc).")
        else:
            st.error("❌ Recommendation: **DECLINE** due to elevated default risk.")

        # Display probability band
        if pred_proba < 0.10:
            band = "Low Risk"
        elif pred_proba < 0.20:
            band = "Medium Risk"
        else:
            band = "High Risk"

        st.write(f"Risk Band: **{band}**")


if __name__ == "__main__":
    main()

