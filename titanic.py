import streamlit as st
import pandas as pd
import pickle

# Load files
model = pickle.load(open('titanic_model.pkl', 'rb'))
scaler = pickle.load(open('titanic_scaler.pkl', 'rb'))
columns = pickle.load(open('titanic_columns.pkl', 'rb'))

st.title("Titanic Survival Prediction")

st.sidebar.header("Passenger Details")

age = st.sidebar.slider("Age", 0, 80, 25)
fare = st.sidebar.number_input("Fare", 0, 512, 32)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
pclass = st.sidebar.selectbox("Ticket Class", ["High (1st)", "Mid (2nd)", "Low (3rd)"])
sibsp = st.sidebar.number_input("Siblings/Spouses", 0, 10, 0)
parch = st.sidebar.number_input("Parents/Children", 0, 10, 0)
has_cabin = st.sidebar.radio("Has Cabin?", ["Yes", "No"])


# encoding
is_female = 1 if gender == "Female" else 0
cabin = 1 if has_cabin == "Yes" else 0

p_high = 1 if pclass == "High (1st)" else 0
p_mid = 1 if pclass == "Mid (2nd)" else 0
p_low = 1 if pclass == "Low (3rd)" else 0


# create dataframe (no need order)
input_df = pd.DataFrame({
    'has_cabin':[cabin],
    'Fare':[fare],
    'Pclass_High':[p_high],
    'Pclass_Mid':[p_mid],
    'Pclass_Low':[p_low],
    'SibSp':[sibsp],
    'Parch':[parch],
    'is_Female':[is_female]
})


# match training column order automatically
input_df = input_df.reindex(columns=columns)


# scale automatically
input_scaled = scaler.transform(input_df)


if st.button("Predict"):
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.success("Survived")
    else:
        st.error("Did Not Survive")
