import streamlit as st
import pandas as pd
import joblib

# -------- Load Saved Objects --------
model = joblib.load("linear_regression.pkl")
scaler = joblib.load("scaler_regression.pkl")
features = joblib.load("features.pkl")   # contains 3 selected features

st.title("🛍️ E-commerce Customer Spending Predictor")
st.write("Predict yearly spending using Linear Regression")

st.header("Enter Customer Details")

# Show inputs ONLY for selected features
input_values = {}

if 'Avg. Session Length' in features:
    input_values['Avg. Session Length'] = st.number_input(
        "Avg. Session Length (30–40)", 30.0, 40.0, 34.0)

if 'Time on App' in features:
    input_values['Time on App'] = st.number_input(
        "Time on App (10–15)", 10.0, 15.0, 12.0)

if 'Length of Membership' in features:
    input_values['Length of Membership'] = st.number_input(
        "Length of Membership (1–6)", 1.0, 6.0, 3.0)

# Create dataframe in correct order
input_df = pd.DataFrame([[input_values[col] for col in features]], columns=features)

# -------- Prediction --------
if st.button("Predict Spending 💰"):
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    st.success(f"Estimated Yearly Spending: **${prediction[0]:.2f}**")
