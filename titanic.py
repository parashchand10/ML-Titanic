import streamlit as st
import pandas as pd
import pickle
import numpy as np

# 1. Load the saved model, scaler, and columns
model = pickle.load(open('titanic_model.pkl', 'rb'))
scaler = pickle.load(open('titanic_scaler.pkl', 'rb'))
columns = pickle.load(open('titanic_columns.pkl', 'rb'))

# --- UI Config ---
st.set_page_config(page_title="Titanic Predictor Pro", layout="wide")

# --- Advanced Custom CSS for Modern UI ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background: linear-gradient(to right, #1e3c72, #2a5298);
        color: white;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Button Styling */
    div.stButton > button:first-child {
        background: linear-gradient(45deg, #00d2ff, #3a7bd5);
        color: white;
        border: none;
        padding: 15px 30px;
        font-size: 20px;
        font-weight: bold;
        border-radius: 50px;
        transition: 0.3s all ease;
        box-shadow: 0 4px 15px rgba(0, 210, 255, 0.3);
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 210, 255, 0.5);
        color: white;
    }

    /* Result Cards */
    .result-card {
        padding: 40px;
        border-radius: 20px;
        text-align: center;
        margin-top: 20px;
        backdrop-filter: blur(15px);
        border: 2px solid rgba(255, 255, 255, 0.1);
    }
    .survived-card {
        background: rgba(46, 213, 115, 0.2);
        border-color: #2ed573;
        color: #2ed573;
    }
    .died-card {
        background: rgba(255, 71, 87, 0.2);
        border-color: #ff4757;
        color: #ff4757;
    }
    
    /* Hide Streamlit Menu and Footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- Sidebar Inputs ---
st.sidebar.image("https://img.icons8.com/clouds/200/titanic.png", width=100)
st.sidebar.title("Configuration")

with st.sidebar:
    age = st.slider("Passenger Age", 0, 80, 25)
    fare = st.number_input("Ticket Fare ($)", 0, 512, 32)
    gender = st.selectbox("Gender", ["Male", "Female"])
    pclass = st.selectbox("Travel Class", ["1st - High", "2nd - Mid", "3rd - Low"])
    st.markdown("---")
    sibsp = st.number_input("Siblings/Spouses", 0, 10, 0)
    parch = st.number_input("Parents/Children", 0, 10, 0)
    has_cabin_input = st.radio("Has Cabin Records?", ["No", "Yes"], horizontal=True)

# --- Preprocessing Logic (Sync with Colab Cell 1654) ---
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
    'Pclass_Low': [p_low],
    'SibSp': [sibsp],
    'Parch': [parch],
    'is_Female': [is_female]
})

# Scaling
scaled_features = scaler.transform(input_df[['Age', 'Fare']])
input_df['Age'] = scaled_features[:, 0]
input_df['Fare'] = scaled_features[:, 1]
input_df = input_df.reindex(columns=columns)

# --- Main Page Content ---
st.title("🚢 RMS Titanic")
st.subheader("Survival Probability Intelligence")
st.write("Using a trained Logistic Regression model to predict passenger fate based on historical data.")

st.markdown("### Run Simulation")
if st.button("EXECUTE PREDICTION"):
    prediction = model.predict(input_df)
    prob = model.predict_proba(input_df)[0][1]
    
    if prediction[0] == 1:
        st.markdown(f"""
            <div class="result-card survived-card">
                <h1 style="font-size: 60px;">POSSIBLE SURVIVOR</h1>
                <p style="font-size: 24px;">The model predicts a <b>{prob:.1%}</b> chance of survival.</p>
            </div>
            """, unsafe_allow_html=True)
        st.balloons()
    else:
        st.markdown(f"""
            <div class="result-card died-card">
                <h1 style="font-size: 60px;">UNLIKELY TO SURVIVE</h1>
                <p style="font-size: 24px;">The model predicts only a <b>{prob:.1%}</b> chance of survival.</p>
            </div>
            """, unsafe_allow_html=True)
