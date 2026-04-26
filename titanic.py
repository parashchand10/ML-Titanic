import streamlit as st
import pandas as pd
import pickle

# 1. Load the saved model and scaler
model = pickle.load(open('titanic_model.pkl', 'rb'))
scaler = pickle.load(open('titanic_scaler.pkl', 'rb'))

# --- UI Header ---
st.title("Titanic Survival Predictor")
st.markdown("Enter the passenger details below to see if they would have survived.")

# --- Main Page Input Section (Two Columns) ---
st.header("Passenger Details")

# Create two columns
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 0, 80, 25)
    sex = st.selectbox("Sex", ["Male", "Female"])
    sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
    has_cabin = st.radio("Has a Cabin?", ["No", "Yes"])

with col2:
    fare = st.number_input("Fare (Ticket Price)", 0, 512, 32)
    pclass = st.selectbox("Ticket Class", ["High (1st)", "Mid (2nd)", "Low (3rd)"])
    parch = st.number_input("Parents/Children Aboard", 0, 10, 0)

# --- Preprocessing ---
is_female = 1 if sex == "Female" else 0
cabin_encoded = 1 if has_cabin == "Yes" else 0

p_high = 1 if pclass == "High (1st)" else 0
p_mid = 1 if pclass == "Mid (2nd)" else 0
p_low = 1 if pclass == "Low (3rd)" else 0

# Create DataFrame matching final_df columns from your [Colab Cell 1370]
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

# --- Scaling ---
# Scale Fare using the same scaler from your [Colab Cell 1353]
scaled_vals = scaler.transform(pd.DataFrame({'Age': [age], 'Fare': [fare]}))
input_df['Fare'] = scaled_vals[0][1] 

# --- Prediction Result (Main Area) ---
st.markdown("---")
if st.button("Predict Survival", use_container_width=True):
    prediction = model.predict(input_df)
    
    if prediction[0] == 1:
        st.success(f"The passenger likely **Survived**!")
        st.balloons()
    else:
        st.error(f"The passenger likely **Did Not Survive**.")
