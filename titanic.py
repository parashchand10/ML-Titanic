import streamlit as st
import pandas as pd
import pickle
import numpy as np

# 1. Load the saved model and scaler
# Make sure these files are in the same folder as this script!
model = pickle.load(open('titanic_model.pkl', 'rb'))
scaler = pickle.load(open('titanic_scaler.pkl', 'rb'))

# --- UI Header ---
st.title("Titanic Survival Predictor")
st.markdown("Enter the passenger details below to see if they would have survived.")

# --- Sidebar / Input Section ---
st.sidebar.header("Passenger Details")

age = st.sidebar.slider("Age", 0, 80, 25)
fare = st.sidebar.number_input("Fare (Ticket Price)", 0, 512, 32)
sex = st.sidebar.selectbox("Sex", ["Female", "Male"])
pclass = st.sidebar.selectbox("Ticket Class", ["High (1st)", "Mid (2nd)", "Low (3rd)"])
sibsp = st.sidebar.number_input("Siblings/Spouses Aboard", 0, 8, 0)
parch = st.sidebar.number_input("Parents/Children Aboard", 0, 6, 0)
has_cabin = st.sidebar.radio("Has a Cabin?", ["Yes", "No"])

# --- Preprocessing the Input ---
# Match the notebook mapping: Female = 1, Male = 0
is_female_encoded = 1 if sex == "Female" else 0
cabin_encoded = 1 if has_cabin == "Yes" else 0

pclass_high = 1 if pclass == "High (1st)" else 0
pclass_mid = 1 if pclass == "Mid (2nd)" else 0
pclass_low = 1 if pclass == "Low (3rd)" else 0

# Create the DataFrame
# MUST match the exact features in Cell 1358 of your notebook
input_data = pd.DataFrame({
    'has_cabin': [cabin_encoded],
    'Fare': [float(fare)],
    'Pclass_High': [pclass_high],
    'Pclass_Mid': [pclass_mid],
    'Pclass_Low': [pclass_low],
    'SibSp': [sibsp],
    'Parch': [parch],
    'is_Female': [is_female_encoded],
    'Age': [float(age)]
})

# --- Apply Scaling ---
# The scaler was trained on [Age, Fare]. We must pass both to transform.
temp_scaling_df = input_data[['Age', 'Fare']]
scaled_values = scaler.transform(temp_scaling_df)

input_data['Age'] = scaled_values[0, 0]
input_data['Fare'] = scaled_values[0, 1]

# --- Final Column Ordering ---
# This MUST match the order of X_train.columns from your notebook
feature_order = ['has_cabin', 'Fare', 'Pclass_High', 'Pclass_Mid', 'Pclass_Low', 'SibSp', 'Parch', 'is_Female']
input_final = input_data[feature_order]

# --- Prediction Logic ---
if st.button("Predict Survival"):
    prediction = model.predict(input_final)
    
    st.write("---") 
    if prediction[0] == 1:
        st.success(f"The passenger likely **Survived**")
    else:
        st.error(f"The passenger likely **Did Not Survive**")

st.info("Note: This prediction is based on the Logistic Regression model from your Colab notebook.")
