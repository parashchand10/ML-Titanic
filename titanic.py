import streamlit as st
import pandas as pd
import pickle
import numpy as np

# 1. Load the saved model and scaler
# Ensure these files are uploaded to your Streamlit repository
model = pickle.load(open('titanic_model.pkl', 'rb'))
scaler = pickle.load(open('titanic_scaler.pkl', 'rb'))

# --- UI Layout ---
st.set_page_config(page_title="Titanic Predictor", layout="wide")

st.title("Titanic Survival Predictor")
st.markdown("---")

# --- Sidebar: Filling Details ---
st.sidebar.header("Passenger Details")
st.sidebar.info("Adjust the values below to simulate a passenger.")

age = st.sidebar.slider("Age", 0, 80, 25)
fare = st.sidebar.number_input("Fare (Ticket Price)", 0, 512, 32)
sex = st.sidebar.selectbox("Sex", ["Female", "Male"])
pclass = st.sidebar.selectbox("Ticket Class", ["High (1st)", "Mid (2nd)", "Low (3rd)"])
sibsp = st.sidebar.number_input("Siblings/Spouses Aboard", 0, 10, 0)
parch = st.sidebar.number_input("Parents/Children Aboard", 0, 10, 0)
has_cabin = st.sidebar.radio("Has a Cabin?", ["No", "Yes"])

# --- Preprocessing Logic (Sync with Colab Cell 1347 & 1358) ---
# Map Gender to is_Female (Female = 1, Male = 0)
is_female = 1 if sex == "Female" else 0
cabin_encoded = 1 if has_cabin == "Yes" else 0

# Map Pclass to One-Hot Dummies
p_high = 1 if pclass == "High (1st)" else 0
p_mid = 1 if pclass == "Mid (2nd)" else 0
p_low = 1 if pclass == "Low (3rd)" else 0

# --- Right Side: Prediction Section ---
# Create a DataFrame with the EXACT column order from your X_train (Cell 1370)
# Order: ['has_cabin', 'Fare', 'Pclass_High', 'Pclass_Mid', 'Pclass_Low', 'SibSp', 'Parch', 'is_Female']
input_dict = {
    'has_cabin': [cabin_encoded],
    'Fare': [fare],
    'Pclass_High': [p_high],
    'Pclass_Mid': [p_mid],
    'Pclass_Low': [p_low],
    'SibSp': [sibsp],
    'Parch': [parch],
    'is_Female': [is_female]
}
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
        st.success(f"### Result: Likely Survived!")
    else:
        st.error(f"### Result: Likely Did Not Survive")

# --- Footer ---
st.info("Note: This prediction is based on a Logistic Regression model trained on the Titanic dataset.")
