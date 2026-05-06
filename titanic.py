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
st.sidebar.header("Passenger Details")

age = st.sidebar.slider("Age", 0, 80, 25)
fare = st.sidebar.number_input("Fare (Ticket Price)", 0, 512, 32)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
pclass = st.sidebar.selectbox("Ticket Class", ["High (1st)", "Mid (2nd)", "Low (3rd)"])
embarked = st.sidebar.selectbox("Port of Embarkation", ["Southampton", "Cherbourg", "Queenstown"])
sibsp = st.sidebar.number_input("Siblings/Spouses Aboard", 0, 10, 0)
parch = st.sidebar.number_input("Parents/Children Aboard", 0, 10, 0)
has_cabin = st.sidebar.radio("Has a Cabin?", ["Yes", "No"])

# --- Preprocessing (Aligned with Colab Cell [199]) ---
is_female = 1 if gender == "Female" else 0
cabin_encoded = 1 if has_cabin == "Yes" else 0

# Pclass Logic
p_high = 1 if pclass == "High (1st)" else 0
p_mid = 1 if pclass == "Mid (2nd)" else 0
p_low = 1 if pclass == "Low (3rd)" else 0

# Embarked Logic (C=0, Q=1, S=2 based on LabelEncoder in Cell [166])
emb_map = {"Cherbourg": 0, "Queenstown": 1, "Southampton": 2}
emb_encoded = emb_map[embarked]

# Create the DataFrame with ALL features in any order first
input_data = {
    'Age': age,
    'has_cabin': cabin_encoded,
    'Fare': fare,
    'Pclass_High': p_high,
    'Pclass_Mid': p_mid,
    'Pclass_Low': p_low,
    'Embarked': emb_encoded,
    'SibSp': sibsp,
    'Parch': parch,
    'is_Female': is_female
}

input_df = pd.DataFrame([input_data])

# Apply Scaling to Age and Fare (Matches Cell [201])
input_df[['Age', 'Fare']] = scaler.transform(input_df[['Age', 'Fare']])

# IMPORTANT: Reindex to match the EXACT column order from X_train (Matches Cell [202])
# X_train order: ['Age', 'has_cabin', 'Fare', 'Pclass_High', 'Pclass_Mid', 'Pclass_Low', 'Embarked', 'SibSp', 'Parch', 'is_Female']
input_df = input_df[columns]

# --- Prediction Result ---
st.subheader("Final Prediction")

if st.button("Predict Survival", use_container_width=True):
    prediction = model.predict(input_df)
    
    if prediction[0] == 1:
        st.markdown(f"""
            <div style="text-align: center; padding: 20px;">
                <h1 style="color: #28a745; font-size: 50px; font-weight: 900; margin: 0;">SURVIVED</h1>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div style="text-align: center; padding: 20px;">
                <h1 style="color: #dc3545; font-size: 50px; font-weight: 900; margin: 0;">NOT SURVIVED</h1>
            </div>
            """, unsafe_allow_html=True)

st.info("Note: This prediction is based on the KNN model saved from the Titanic Dataset notebook.")
