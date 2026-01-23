import streamlit as st
import joblib
import numpy as np
import pandas as pd

MODEL_PATH = "artifacts/credit_default_model.joblib"

@st.cache_resource
def load_model():
    payload = joblib.load(MODEL_PATH)
    return payload["model"], payload["numeric_features"], payload["categorical_features"]

model, numeric_feats, categorical_feats = load_model()

st.title("📊 Credit Risk Default Model")
st.write("Predict whether a loan is likely to default based on borrower features.")

st.sidebar.header("Loan Application")

user_input = {}
default_values = {
    "loan_amnt": 15000.0,
    "int_rate": 12.5,
    "annual_inc": 60000.0,
    "dti": 18.5,
    "revol_util": 45.0,
    "open_acc": 8,
    "total_acc": 20,
    "term": "36 months",
    "grade": "C",
    "home_ownership": "MORTGAGE",
    "purpose": "debt_consolidation",
}

for col in numeric_feats:
    user_input[col] = st.sidebar.number_input(
    col,
    value=float(default_values.get(col, 0.0))
)


for col in categorical_feats:
    options_map = {
    "term": ["36 months", "60 months"],
    "grade": ["A","B","C","D","E","F","G"],
    "home_ownership": ["MORTGAGE","RENT","OWN","OTHER","NONE"],
    "purpose": [
        "debt_consolidation","credit_card","home_improvement",
        "small_business","major_purchase","car","other"
    ],
}

if col in options_map:
    opts = options_map[col]
    default = default_values.get(col, opts[0])
    user_input[col] = st.sidebar.selectbox(col, opts, index=opts.index(default))
else:
    user_input[col] = st.sidebar.text_input(col, value="")


if st.sidebar.button("Predict Default Risk"):
    df_input = pd.DataFrame([user_input])
    prediction = model.predict(df_input)[0]
    proba = model.predict_proba(df_input)[0][1]

    st.subheader("🧾 Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ High Risk of Default (Probability: {proba:.2f})")
    else:
        st.success(f"✅ Low Risk of Default (Probability: {proba:.2f})")
