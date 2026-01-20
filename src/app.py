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

# Build dynamic form from expected features
user_input = {}

for col in numeric_feats:
    user_input[col] = st.sidebar.number_input(col, value=0.0)

for col in categorical_feats:
    user_input[col] = st.sidebar.text_input(col, value="")

# Predict
if st.sidebar.button("Predict Default Risk"):
    df_input = pd.DataFrame([user_input])
    prediction = model.predict(df_input)[0]
    proba = model.predict_proba(df_input)[0][1]

    st.subheader("🧾 Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ High Risk of Default (Probability: {proba:.2f})")
    else:
        st.success(f"✅ Low Risk of Default (Probability: {proba:.2f})")

