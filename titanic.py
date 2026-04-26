import streamlit as st
import pandas as pd
import pickle
import numpy as np

# 1. Load the saved model, scaler, and columns from your Colab
model = pickle.load(open('titanic_model.pkl', 'rb'))
scaler = pickle.load(open('titanic_scaler.pkl', 'rb'))
columns = pickle.load(open('titanic_columns.pkl', 'rb'))

# --- Page Config ---
st.set_page_config(page_title="Titanic Predictor", layout="centered")

# --- Simple & Attractive CSS ---
st.markdown("""
    <style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
        background-color: #f8f9fa;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 2rem;
        max-width: 700px;
    }

    /* Clean Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e9ecef;
    }

    /* Minimalist Button */
    div.stButton > button:first-child {
        background-color: #007bff;
        color: white;
        border: none;
        padding: 12px 24px;
        font-size: 16px;
        font-weight: 600;
        border-radius: 8px;
        width: 100%;
        transition: all 0.2s ease;
    }
    div.stButton > button:hover {
        background-color: #0056b3;
        border: none;
        color: white;
    }

    /* Result Cards */
    .result-box {
        padding: 30px;
        border-radius: 12px;
        text-align: center;
        margin-top: 25px;
        border: 1px solid #e0e0e0;
    }
    .survived {
        background-color: #e7f4e9;
        color: #1e7e34;
        border-color: #c3e6cb;
    }
    .not-survived {
        background-color: #fce8e8;
        color: #d93025;
        border-color: #f5c6cb;
    }
    </style>
    """, unsafe_allow_html=True)

# --- App Header ---
st.title("🚢 Titanic Survival AI")
st.text("Predict passenger outcomes using machine learning.")
st.markdown("---")

# --- Sidebar Inputs ---
st.sidebar.header("Passenger Profile")
with st.sidebar:
    age = st.slider("Age", 0, 80, 25)
    fare = st.number_input("Ticket Fare", 0, 512, 32)
    gender = st.selectbox("Gender", ["Male", "Female"])
    pclass = st.selectbox("Class", ["1st Class", "2nd Class", "3rd Class"])
    st.markdown("---")
    sibsp = st.number_input("Siblings/Spouses", 0, 8, 0)
    parch = st.number_input("Parents/Children", 0, 6, 0)
    has_cabin_input = st.radio("Cabin Records?", ["No", "Yes"], horizontal=True)

# --- Preprocessing (Sync with Colab Cell 1654) ---
is_female = 1 if gender == "Female" else 0
cabin = 1 if has_cabin_input == "Yes" else 0
p_high = 1 if "1st" in pclass else 0
p_mid = 1 if "2nd" in pclass else 0
p_low = 1 if "3rd" in pclass else 0

input_df = pd.DataFrame({
    'Age': [age],
    'has_cabin': [cabin],
    'Fare': [fare],
    'Pclass_High': [p_high],
    'Pclass_Mid': [p_mid],
