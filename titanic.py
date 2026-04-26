import streamlit as st
import pandas as pd
import pickle
import numpy as np

# 1. Load the saved model, scaler, and columns
model = pickle.load(open('titanic_model.pkl', 'rb'))
scaler = pickle.load(open('titanic_scaler.pkl', 'rb'))
columns = pickle.load(open('titanic_columns.pkl', 'rb'))

# --- Simple & Attractive CSS ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button {
        background-color: #add8e6;
        color: black;
        border-radius: 10px;
        font-weight: bold;
        width: 100%;
        border: none;
        padding: 10px;
    }
    .result-text {
        font-size: 45px !important;
        font-weight: bold;
        text-align: center;
        padding: 20px;
        border-radius: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🚢 Titanic Survival Prediction")

# --- Sidebar Inputs ---
st.sidebar.header("Passenger Details")
age = st.sidebar.slider("Age", 0, 80, 25)
fare = st.sidebar.number_input("Fare", 0, 512, 32)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
pclass = st.sidebar.selectbox("Ticket Class", ["High (1st)", "Mid (2nd)", "Low (3rd)"])
sibsp = st.sidebar.number_input("Siblings/Spouses", 0, 10, 0)
parch = st.sidebar.number_input("Parents/Children", 0, 10, 0)
has_cabin_input = st.sidebar.radio("Has Cabin?", ["No", "Yes"])

# --- Preprocessing (Matching your Colab logic) ---
is_female = 1 if gender == "Female" else 0
cabin = 1 if has_cabin_input == "Yes" else 0
p_high = 1 if "High" in pclass else 0
p_mid = 1 if "Mid" in pclass else 0
p_low = 1 if "3rd" in pclass else 0

# FIXED DATAFRAME BLOCK
input_df = pd.DataFrame({
    'Age': [age],
    'has_cabin': [cabin],
    'Fare': [fare],
    'Pclass_High': [p_high],
    'Pclass_Mid': [p_mid],
    'Pclass_Low': [p_low],
    'SibSp': [sibsp],
    'Parch': [parch],
    'is_Female': [is_female]
})

# Scaling
scaled_features = scaler.transform(input_df[['Age', 'Fare']])
input_df['Age'] = scaled_features[:, 0]
input_df['Fare'] = scaled_features[:, 1]

# Reorder to match your [Colab Cell 1672](https://colab.research.google.com/drive/14kxGJG_YCa1Df2NIw-KUqAJ2PVLqqzbK#scrollTo=oaPLhu5GCU6b)
input_df = input_df.reindex(columns=columns)

# --- Prediction Result ---
st.subheader("Prediction Result")
if st.button("Predict Survival"):
    prediction = model.predict(input_df)
    
    if prediction[0] == 1:
        st.markdown('<div style="background-color:#d4edda; color:#155724;" class="result-text">✨ Survived!</div>', unsafe_allow_html=True)
        st.balloons()
    else:
        st.markdown('<div style="background-color:#f8d7da; color:#721c24;" class="result-text">💀 Did Not Survive</div>', unsafe_allow_html=True)
