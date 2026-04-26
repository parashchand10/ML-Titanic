import streamlit as st
import pandas as pd
import pickle
import numpy as np

# 1. Load the saved model, scaler, and columns from your Colab
model = pickle.load(open('titanic_model.pkl', 'rb'))
scaler = pickle.load(open('titanic_scaler.pkl', 'rb'))
columns = pickle.load(open('titanic_columns.pkl', 'rb'))

# --- UI Header ---
st.title("Titanic Survival Prediction")

# --- Sidebar / Input Section ---
st.sidebar.title("Passenger Details")
st.sidebar.header("Enter the passenger details below to see if they would have survived the disaster.")

age = st.sidebar.slider("Age", 0, 80, 25)
fare = st.sidebar.number_input("Fare (Ticket Price)", 0, 512, 32)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
pclass = st.sidebar.selectbox("Ticket Class", ["High (1st)", "Mid (2nd)", "Low (3rd)"])
sibsp = st.sidebar.number_input("Siblings/Spouses Aboard", 0, 10, 0)
parch = st.sidebar.number_input("Parents/Children Aboard", 0, 10, 0)
has_cabin = st.sidebar.radio("Has a Cabin?", ["Yes", "No"])

# --- Preprocessing (Sync with Colab Cell 1672) ---
is_female = 1 if gender == "Female" else 0
cabin_encoded = 1 if has_cabin == "Yes" else 0

p_high = 1 if pclass == "High (1st)" else 0
p_mid = 1 if pclass == "Mid (2nd)" else 0
p_low = 1 if pclass == "Low (3rd)" else 0

# 2. Create the DataFrame with ALL features including Age
# Note: Age MUST be here because your scaler and model expect it
input_df = pd.DataFrame({
    'Age': [age],
    'has_cabin': [cabin_encoded],
    'Fare': [fare],
    'Pclass_High': [p_high],
    'Pclass_Mid': [p_mid],
    'Pclass_Low': [p_low],
    'SibSp': [sibsp],
    'Parch': [parch],
    'is_Female': [is_female]
})

# 3. Apply Scaling to Age and Fare
# transform expects a 2D array of the numeric columns
input_df[['Age', 'Fare']] = scaler.transform(input_df[['Age', 'Fare']])

# 4. Ensure the column order matches your training exactly
input_df = input_df.reindex(columns=columns)

# --- Prediction Result ---
st.subheader("Final Prediction")

if st.button("Predict Survival", use_container_width=True):
    prediction = model.predict(input_df)
    
    if prediction[0] == 1:
        st.success(f"### Result: Survived!")
        st.balloons()
    else:
        st.error(f"### Result: Did Not Survive")

st.info("Note: This prediction is based on the Logistic Regression model from your Titanic Dataset notebook.")
