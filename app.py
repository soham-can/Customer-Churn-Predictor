import streamlit as st
import pandas as pd
import joblib

model = joblib.load('models/xgb_churn_model.pkl')
feature_columns = joblib.load('models/feature_columns.pkl')
st.write("Click below for list all features",feature_columns)
st.title("Customer Churn Predictor")

# Basic inputs - adjust based on your actual columns
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", 18.0, 120.0, 70.0)
total_charges = st.number_input("Total Charges", 0.0, 9000.0, 1000.0)
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
# ... add remaining inputs for your other columns

if st.button("Predict"):
    # Build input dict matching your training columns
    input_dict = {col: 0 for col in feature_columns}
    input_dict['tenure'] = tenure
    input_dict['MonthlyCharges'] = monthly_charges
    input_dict['TotalCharges'] = total_charges
    # set one-hot columns based on selections, e.g.:
    if f'Contract_{contract}' in input_dict:
        input_dict[f'Contract_{contract}'] = 1
    if f'InternetService_{internet_service}' in input_dict:
        input_dict[f'InternetService_{internet_service}'] = 1

    input_df = pd.DataFrame([input_dict])[feature_columns]
    prediction = model.predict(input_df)[0]
    st.write("Churn Prediction:", "Yes" if prediction == 1 else "No")
