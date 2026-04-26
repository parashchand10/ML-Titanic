import streamlit as st
import pandas as pd
import pickle
import numpy as np

# 1. Load the saved model and scaler
model = pickle.load(open('titanic_model.pkl', 'rb'))
scaler = pickle.load(open('titanic_scaler.pkl', 'rb'))

# --- UI Header ---
st.title("Titanic Survival Prediction")
st.markdown("Enter the passenger details below to see if they would have survived the disaster.")

# --- Sidebar / Input Section ---
st.sidebar.header("Passenger Details")

# User Inputs
age = st.sidebar.slider("Age", 0, 80, 25)
fare = st.sidebar.number_input("Fare (Ticket Price)", 0, 512, 32)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
pclass = st.sidebar.selectbox("Ticket Class", ["High (1st)", "Mid (2nd)", "Low (3rd)"])
sibsp = st.sidebar.number_input("Siblings/Spouses Aboard", 0, 10, 0)
parch = st.sidebar.number_input("Parents/Children Aboard", 0, 10, 0)
has_cabin = st.sidebar.radio("Has a Cabin?", ["Yes", "No"])

# --- Preprocessing the Input ---
# Convert inputs to match training data format
# Map Gender to is_Female (Female = 1, Male = 0)
is_female = 1 if gender == "Female" else 0
cabin_encoded = 1 if has_cabin == "Yes" else 0

# Map Pclass to One-Hot Dummies
p_high = 1 if pclass == "High (1st)" else 0
p_mid = 1 if pclass == "Mid (2nd)" else 0
p_low = 1 if pclass == "Low (3rd)" else 0

# Create a DataFrame for the input
# Note: The order must match X.columns from your training exactly!
input_dict = pd.DataFrame({
    'has_cabin': [cabin_encoded],
    'Fare': [fare],
    'Pclass_High': [p_high],
    'Pclass_Mid': [p_mid],
    'Pclass_Low': [p_low],
    'SibSp': [sibsp],
    'Parch': [parch],
    'is_Female': [is_female]
})

input_df = pd.DataFrame(input_dict)

# Apply Scaling to Fare (The scaler was fit on [Age, Fare])
# Even if Age isn't a feature, the scaler expects it for the transformation
scaled_vals = scaler.transform(pd.DataFrame({'Age': [age], 'Fare': [fare]}))
input_df['Fare'] = scaled_vals[0][1] 

st.subheader("Final Prediction")
st.write("Click the button below to run the Logistic Regression model based on your sidebar inputs.")

if st.button("Predict Survival", use_container_width=True):
    # Ensure columns match training order exactly
    prediction = model.predict(input_df)
    
    if prediction[0] == 1:
        st.success(f"### Result: Survived!")
    else:
        st.error(f"### Result: Did Not Survive")

# --- Footer ---
st.info("Note: This prediction is based on a Logistic Regression model trained on the Titanic dataset.")
