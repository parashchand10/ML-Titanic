import streamlit as st
import pandas as pd
import pickle
import numpy as np

# 1. Load the saved model and scaler
# Ensure these files are in the same folder as this script in your GitHub repo
model = pickle.load(open('titanic_model.pkl', 'rb'))
scaler = pickle.load(open('titanic_scaler.pkl', 'rb'))

# --- UI Header ---
st.title("🚢 Titanic Survival Predictor")
st.markdown("Enter the passenger details below to see if they would have survived the disaster.")

# --- Sidebar / Input Section ---
st.sidebar.header("Passenger Details")

# User Inputs
age = st.sidebar.slider("Age", 0, 80, 25)
fare = st.sidebar.number_input("Fare (Ticket Price)", 0, 512, 32)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
pclass = st.sidebar.selectbox("Ticket Class", ["High (1st)", "Mid (2nd)", "Low (3rd)"])
has_cabin = st.sidebar.radio("Has a Cabin?", ["Yes", "No"])

# --- Preprocessing the Input ---
# Convert inputs to match training data format
sex_encoded = 1 if sex == "Male" else 0
cabin_encoded = 1 if has_cabin == "Yes" else 0

# Handle Pclass Dummies
pclass_high = 1 if pclass == "High (1st)" else 0
pclass_mid = 1 if pclass == "Mid (2nd)" else 0
pclass_low = 1 if pclass == "Low (3rd)" else 0

# Create the initial DataFrame
input_data = pd.DataFrame({
    'has_cabin': [cabin_encoded],
    'Fare': [fare],
    'Pclass_High': [pclass_high],
    'Pclass_Mid': [pclass_mid],
    'Pclass_Low': [pclass_low],
    'Sex': [sex_encoded],
    'Age': [age] 
})

# Reorder columns to match the model's training order exactly
# Note: Keeping Age out of the final features if it was dropped during training
feature_columns = ["has_cabin", "Fare", "Pclass_High", "Pclass_Mid", "Pclass_Low", "Sex"]
input_final = input_data[feature_columns].copy()

# --- FIX: Apply Scaling to Fare ---
# The scaler expects 'Age' and 'Fare' as input based on your previous error
temp_scaling_df = pd.DataFrame({'Age': [age], 'Fare': [fare]})
scaled_array = scaler.transform(temp_scaling_df)

# Assign the scaled Fare value (index 1 of the returned array) to our input
input_final.loc[0, 'Fare'] = scaled_array[0, 1]

# --- Prediction Logic ---
if st.button("Predict Survival"):
    # Ensure input_final is shaped correctly for the model
    prediction = model.predict(input_final)
    probability = model.predict_proba(input_final)[0][1]
    
    st.write("---") # Visual separator
    if prediction[0] == 1:
        st.success(f"✨ The passenger likely **Survived**! (Probability: {probability:.2%})")
        st.balloons()
    else:
        st.error(f"💀 The passenger likely **Did Not Survive**. (Probability of survival: {probability:.2%})")

# --- Footer ---
st.info("Note: This prediction is based on a Logistic Regression model trained on the Titanic dataset.")
