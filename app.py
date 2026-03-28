import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection — Elvis Osinachi",
    page_icon="🛡️",
    layout="centered"
)

# ── Load model ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# ── Header ───────────────────────────────────────────────────
st.title("🛡️ Mobile Money Fraud Detector")
st.markdown(
    "XGBoost model trained on mobile payment transactions. "
    "Achieves **99% recall** at 0.9 threshold — built by "
    "[Elvis Osinachi](https://ifeanyiosinachi.vercel.app)."
)
st.divider()

# ── Inputs ───────────────────────────────────────────────────
st.subheader("Transaction Details")

col1, col2 = st.columns(2)

with col1:
    tx_type = st.selectbox(
        "Transaction Type",
        ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
    )
    amount = st.number_input("Amount (₦)", min_value=0.0, value=5000.0, step=100.0)
    oldbalanceOrg = st.number_input("Sender Opening Balance (₦)", min_value=0.0, value=10000.0, step=100.0)
    newbalanceOrig = st.number_input("Sender Closing Balance (₦)", min_value=0.0, value=5000.0, step=100.0)

with col2:
    oldbalanceDest = st.number_input("Receiver Opening Balance (₦)", min_value=0.0, value=0.0, step=100.0)
    newbalanceDest = st.number_input("Receiver Closing Balance (₦)", min_value=0.0, value=5000.0, step=100.0)

# ── Feature engineering (must match notebook) ────────────────
balanceDiffOrg = oldbalanceOrg - newbalanceOrig
balanceDiffDest = newbalanceDest - oldbalanceDest

input_df = pd.DataFrame([{
    "type": tx_type,
    "amount": amount,
    "oldbalanceOrg": oldbalanceOrg,
    "newbalanceOrig": newbalanceOrig,
    "oldbalanceDest": oldbalanceDest,
    "newbalanceDest": newbalanceDest,
    "balanceDiffOrg": balanceDiffOrg,
    "balanceDiffDest": balanceDiffDest,
}])

st.divider()

# ── Predict ──────────────────────────────────────────────────
if st.button("🔍 Analyse Transaction", use_container_width=True):
    proba = model.predict_proba(input_df)[0][1]
    threshold = 0.9
    prediction = int(proba >= threshold)

    st.subheader("Result")

    if prediction == 1:
        st.error(f"⚠️ **FRAUDULENT** — Fraud probability: `{proba:.1%}`")
        st.markdown(
            "This transaction has been flagged as likely fraudulent based on "
            "balance patterns and transaction type."
        )
    else:
        st.success(f"✅ **LEGITIMATE** — Fraud probability: `{proba:.1%}`")
        st.markdown("This transaction appears legitimate.")

    # Show feature breakdown
    with st.expander("See input features"):
        st.dataframe(input_df, use_container_width=True)

st.divider()
st.caption(
    "Model: XGBoost + SMOTE | ROC-AUC: 0.999 | Recall: 99% | Precision: 51% @ threshold 0.9 | "
    "GitHub: [0sinach1](https://github.com/0sinach1)"
)
