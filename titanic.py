import streamlit as st
import pandas as pd
import pickle
import numpy as np

# 1. Load the saved model and scaler
model = pickle.load(open('titanic_model.pkl', 'rb'))
scaler = pickle.load(open('titanic_scaler.pkl', 'rb'))

# --- UI Layout Config ---
st.set_page_config(page_title="Titanic Predictor", layout="wide")

# --- Custom CSS: Lock Sidebar & Style Results ---
st.markdown("""
    <style>
    /* 1. Lock Sidebar - prevents internal scrolling if content fits */
    [data-testid="stSidebar"] {
        overflow: hidden;
    }
    
    /* 2. Light Blue Button Styling */
    div.stButton > button:first-child {
        background-color: #add8e6;
        color: black;
        border: none;
        height: 3em;
        width: 100%;
        border-radius: 5px;
        font-weight: bold;
    }
    div.stButton > button:hover {
        background-color: #87ceeb;
    }

    /* 3. Result Font Size Styling */
    .result-text {
        font-size: 45px !important;
        font-weight: bold;
        text-align: center;
        padding: 20px;
        border-radius: 10px;
    }
    .survived { background-color: #d4edda; color: #155724; }
    .died { background-color: #f8d7da; color: #721c24; }
    </style>
    """, unsafe_allow_html=True)

st.title("Titanic Survival Prediction")
st.markdown("Enter the passenger details below to see if they would have survived the disaster.")

# --- Sidebar: Passenger Details ---
st.sidebar.header("Passenger Details")

age = st.sidebar.slider("Age", 0, 80, 25)
fare = st.sidebar.number_input("Fare (Ticket Price)", 0, 512, 32)
sex = st.sidebar.selectbox("Gender", ["Female", "Male"])
pclass = st.sidebar.selectbox("Ticket Class", ["High (1st)", "Mid (2nd)", "Low (3rd)"])
sibsp = st.sidebar.number_input("Siblings/Spouses Aboard", 0, 10, 0)
parch = st.sidebar.number_input("Parents/Children Aboard", 0, 10, 0)
has_cabin = st.sidebar.radio("Has a Cabin?", ["No", "Yes"])

# --- Preprocessing ---
is_female = 1 if sex == "Female" else 0
cabin_encoded = 1 if has_cabin == "Yes" else 0
p_high, p_mid, p_low = (1,0,0) if "High" in pclass else (0,1,0) if "Mid" in pclass else (0,0,1)

input_df = pd.DataFrame({
    'has_cabin': [cabin_encoded],
    'Fare': [fare],
    'Pclass_High': [p_high],
    'Pclass_Mid': [p_mid],
    'Pclass_Low': [p_low],
    'SibSp': [sibsp],
    'Parch': [parch],
    'is_Female': [is_female]
})

# Scaling (Sync with your [Colab Cell 1374](https://colab.research.google.com/drive/14kxGJG_YCa1Df2NIw-KUqAJ2PVLqqzbK#scrollTo=oaPLhu5GCU6b))
scaled_vals = scaler.transform(pd.DataFrame({'Age': [age], 'Fare': [fare]}))
input_df['Fare'] = scaled_vals[0][1] 

# --- Prediction Logic ---
st.subheader("Final Prediction")

if st.button("Predict Survival", use_container_width=True):
 if st.button("Predict Survival", use_container_width=True):
    prediction = model.predict(input_df)
    
    if prediction[0] == 1:
        st.markdown('''
            <div class="result-container">
                <div class="result-text survived">✨ Result: Likely Survived!</div>
            </div>
            ''', unsafe_allow_html=True)
        st.balloons()
    else:
        st.markdown('''
            <div class="result-container">
                <div class="result-text died">💀 Result: Did Not Survive</div>
            </div>
            ''', unsafe_allow_html=True)
st.markdown("---")
st.info("Note: This prediction is based on the Logistic Regression model from your Titanic Dataset notebook.")
