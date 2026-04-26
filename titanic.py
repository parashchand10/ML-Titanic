import streamlit as st
import pandas as pd
import pickle
import numpy as np

# 1. Load the saved model and scaler
# Make sure these files are in the same folder as this script
model = pickle.load(open('titanic_model.pkl', 'rb'))
scaler = pickle.load(open('titanic_scaler.pkl', 'rb'))

# --- UI Header ---
st.title("Titanic Survival Predictor")
st.markdown("Enter the passenger details below to see if they would have survived.")

# --- Main Page Input Section (Single Column) ---
st.sidebar.header("Passenger Details")

age = st.slider("Age", 0, 80, 25)
fare = st.number_input("Fare (Ticket Price)", 0, 512, 32)
sex = st.selectbox("Sex", ["Male", "Female"])
pclass = st.selectbox("Ticket Class", ["High (1st)", "Mid (2nd)", "Low (3rd)"])
sibsp = st.number_input("Siblings/Spouses Aboard (SibSp)", 0, 10, 0)
parch = st.number_input("Parents/Children Aboard (Parch)", 0, 10, 0)
has_cabin = st.radio("Has a Cabin?", ["Yes", "No"], index=1)

# --- Preprocessing ---
# Match the logic from your Colab [Cell 1347]
is_female = 1 if sex == "Female" else 0
cabin_encoded = 1 if has_cabin == "Yes" else 0

# Match Pclass Dummies from [Cell 1351]
p_high = 1 if pclass == "High (1st)" else 0
p_mid = 1 if pclass == "Mid (2nd)" else 0
p_low = 1 if pclass == "Low (3rd)" else 0

# Create DataFrame with exact column order from final_df [Cell 1370]
# Order: [has_cabin, Fare, Pclass_High, Pclass_Mid, Pclass_Low, SibSp, Parch, is_Female]
# Note: Age is used for scaling but was NOT in your final X features in Cell 1370
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
# We must scale Age and Fare using the training scaler [Cell 1353]
# Even if Age isn't a feature, the scaler expects it in the transform call
scaled_vals = scaler.transform(pd.DataFrame({'Age': [age], 'Fare': [fare]}))
input_df['Fare'] = scaled_vals[0][1] 

# --- Prediction ---
st.markdown("---")
if st.button("Predict Survival", use_container_width=True):
    # Ensure column order matches X.columns perfectly
    prediction = model.predict(input_df)
    
    if prediction[0] == 1:
        st.success(f"The passenger likely **Survived**!")
        st.balloons()
    else:
        st.error(f"The passenger likely **Did Not Survive**.")

st.info("Note: This prediction is based on the Logistic Regression model from your Colab notebook.")
