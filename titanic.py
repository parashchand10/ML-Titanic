import streamlit as st
import pandas as pd
import pickle

# load
model = pickle.load(open('titanic_model.pkl', 'rb'))
scaler = pickle.load(open('titanic_scaler.pkl', 'rb'))
columns = pickle.load(open('titanic_columns.pkl', 'rb'))

st.title("Titanic Survival Prediction")

age = st.sidebar.slider("Age", 0, 80, 25)
fare = st.sidebar.number_input("Fare", 0, 512, 32)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
pclass = st.sidebar.selectbox("Ticket Class", ["High (1st)", "Mid (2nd)", "Low (3rd)"])
sibsp = st.sidebar.number_input("Siblings", 0, 10, 0)
parch = st.sidebar.number_input("Parents", 0, 10, 0)
has_cabin = st.sidebar.radio("Has Cabin?", ["Yes", "No"])

# encoding
is_female = 1 if gender == "Female" else 0
cabin = 1 if has_cabin == "Yes" else 0

p_high = 1 if pclass == "High (1st)" else 0
p_mid = 1 if pclass == "Mid (2nd)" else 0
p_low = 1 if pclass == "Low (3rd)" else 0

# dataframe
input_df = pd.DataFrame({
    'Age':[age],
    'Fare':[fare],
    'has_cabin':[cabin],
    'Pclass_High':[p_high],
    'Pclass_Mid':[p_mid],
    'Pclass_Low':[p_low],
    'SibSp':[sibsp],
    'Parch':[parch],
    'is_Female':[is_female]
})

# scale only Age & Fare
input_df[['Age','Fare']] = scaler.transform(
    input_df[['Age','Fare']].values
)

# reorder
input_df = input_df.reindex(columns=columns)

# predict
if st.button("Predict"):
    prediction = model.predict(input_df)

    if prediction[0] == 1:
        st.success("Survived")
    else:
        st.error("Did Not Survive")
