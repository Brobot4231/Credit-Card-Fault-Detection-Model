import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved Random Forest model
model = joblib.load("random_forest_fraud_model.pkl")

# App Title
st.title("💳 Credit Card Fraud Detection")
st.write("Upload transaction data OR enter a single transaction to get fraud predictions with confidence scores.")

# ================================
# Option 1: Upload CSV File
# ================================
st.header("📂 Batch Prediction (Upload CSV)")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.write(data.head())

    required_cols = [f"V{i}" for i in range(1,29)] + ["Time","Amount"]
    if all(col in data.columns for col in required_cols):
        predictions = model.predict(data)
        probabilities = model.predict_proba(data)[:,1]

        data["Prediction"] = predictions
        data["Confidence"] = probabilities

        st.subheader("Predictions")
        st.write(data[["Prediction","Confidence"]].head())

        csv = data.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Predictions",
            data=csv,
            file_name="fraud_predictions.csv",
            mime="text/csv",
        )
    else:
        st.error("Uploaded file must contain columns: Time, Amount, V1–V28")

# ================================
# Option 2: Single Transaction Input
# ================================
st.header("📝 Single Transaction Prediction")

with st.form("single_transaction_form"):
    time = st.number_input("Time", value=0.0)
    amount = st.number_input("Amount", value=0.0)

    v_features = []
    for i in range(1,29):
        v_features.append(st.number_input(f"V{i}", value=0.0))

    submitted = st.form_submit_button("Predict")

    if submitted:
        # Create dataframe for single input
        input_data = pd.DataFrame([ [time] + v_features + [amount] ],
                                  columns=["Time"] + [f"V{i}" for i in range(1,29)] + ["Amount"])

        prediction = model.predict(input_data)[0]
        confidence = model.predict_proba(input_data)[0][1]

        st.subheader("🔍 Prediction Result")
        st.write(f"**Prediction:** {'Fraud' if prediction==1 else 'Legit'}")
        st.write(f"**Confidence Score:** {confidence:.4f}")

