import streamlit as st
import pandas as pd
import pickle
import numpy as np

# 1. Load the saved model and scaler
model = pickle.load(open('titanic_model.pkl', 'rb'))
scaler = pickle.load(open('titanic_scaler.pkl', 'rb'))

# --- UI Header ---
st.title("Titanic Survival Predictor")
st.markdown("Enter the passenger details below to see if they would have survived the disaster.")

# --- Sidebar / Input Section ---
st.sidebar.header("Passenger Details")

age = st.sidebar.slider("Age", 0, 80, 25)
fare = st.sidebar.number_input("Fare (Ticket Price)", 0, 512, 32)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
pclass = st.sidebar.selectbox("Ticket Class", ["High (1st)", "Mid (2nd)", "Low (3rd)"])
has_cabin = st.sidebar.radio("Has a Cabin?", ["Yes", "No"])

# --- Preprocessing the Input ---
sex_encoded = 1 if sex == "Male" else 0
cabin_encoded = 1 if has_cabin == "Yes" else 0
pclass_high = 1 if pclass == "High (1st)" else 0
pclass_mid = 1 if pclass == "Mid (2nd)" else 0
pclass_low = 1 if pclass == "Low (3rd)" else 0

# Create the initial DataFrame
input_data = pd.DataFrame({
    'has_cabin': [cabin_encoded],
    'Fare': [float(fare)],  # FORCE Fare to be a float here
    'Pclass_High': [pclass_high],
    'Pclass_Mid': [pclass_mid],
    'Pclass_Low': [pclass_low],
    'Sex': [sex_encoded],
    'Age': [age] 
})

# Reorder columns
feature_columns = ["has_cabin", "Fare", "Pclass_High", "Pclass_Mid", "Pclass_Low", "Sex"]
input_final = input_data[feature_columns].copy()

# --- Apply Scaling to Fare ---
# Creating temp DF for scaler
temp_scaling_df = pd.DataFrame({'Age': [age], 'Fare': [float(fare)]})
scaled_array = scaler.transform(temp_scaling_df)

# Assign the scaled value (Fare is index 1)
# Using .astype(float) on the column ensures it can receive the decimal value
input_final['Fare'] = input_final['Fare'].astype(float)
input_final.loc[0, 'Fare'] = scaled_array[0, 1]

# --- Prediction Logic ---
if st.button("Predict Survival"):
    prediction = model.predict(input_final)
    
    st.write("---") 
    if prediction[0] == 1:
        st.success(f"The passenger likely **Survived**!
        st.balloons()
    else:
        st.error(f"The passenger likely **Did Not Survive**.")

st.info("Note: This prediction is based on a Logistic Regression model trained on the Titanic dataset.")
