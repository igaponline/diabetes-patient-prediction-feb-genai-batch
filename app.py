import pandas as pd
import numpy as np
import streamlit as st
import pickle

st.title("Diabetes Prediction App")

scaler = pickle.load(open("scaler.pkl", "rb"))
model = pickle.load(open("diabetes_xgboost_model.pkl", "rb"))


age = st.sidebar.number_input("Age", min_value=18, max_value=80, value=30)
hypertension = st.sidebar.selectbox("Hypertension", options=[0, 1])
heart_disease = st.sidebar.selectbox("Heart Disease", options=[0, 1])
bmi = st.sidebar.number_input("BMI", min_value=10, max_value=50, value=25)
HbA1c_level = st.sidebar.number_input("HbA1c Level", min_value=0.0, max_value=10.0, value=5.5)
blood_glucose_level = st.sidebar.number_input("Blood Glucose Level", min_value=0, max_value=500, value=100)
gender = st.sidebar.selectbox("Gender", options=["Male", "Female", "Other"])
smoking = st.sidebar.selectbox("Smoking Status", options=["Never", "Past Smoker", "Current", "Unknown"])

if st.sidebar.button("Predict Diabetes"):
    st.write("Input Values:")
    st.write(f"Age: {age}")
    st.write(f"Hypertension: {hypertension}")
    st.write(f"Heart Disease: {heart_disease}")
    st.write(f"BMI: {bmi}")
    st.write(f"HbA1c Level: {HbA1c_level}")
    st.write(f"Blood Glucose Level: {blood_glucose_level}")
    st.write(f"Gender: {gender}")
    st.write(f"Smoking Status: {smoking}")
    gender_Female = 0
    gender_Male = 0
    gender_Other = 0
    if gender == "Male":
        gender_Male = 1
    elif gender == "Female":
        gender_Female = 1
    else:
        gender_Other = 1

    smoking_history_current = 0
    smoking_history_never = 0
    smoking_history_past_smoker = 0
    smoking_history_unknown = 0

    if smoking == "Never":
        smoking_history_never = 1
    elif smoking == "Past Smoker":
        smoking_history_past_smoker = 1
    elif smoking == "Current":
        smoking_history_current = 1
    else:
        smoking_history_unknown = 1
    
    columns = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level',
       'blood_glucose_level', 'gender_Female', 'gender_Male', 'gender_Other',
       'smoking_history_current', 'smoking_history_never',
       'smoking_history_past_smoker', 'smoking_history_unknown']
    data = [[age, hypertension, heart_disease, bmi, HbA1c_level, blood_glucose_level, gender_Female, gender_Male, gender_Other,
       smoking_history_current, smoking_history_never, smoking_history_past_smoker, smoking_history_unknown]]
    
    myinput = pd.DataFrame(data = data, columns = columns)
    scale_cols = ['age','bmi','HbA1c_level','blood_glucose_level']
    myinput[scale_cols] = scaler.transform(myinput[scale_cols])

    # st.write(myinput)
    prediction = model.predict(myinput)
    if prediction[0] == 1:
        st.error("The model predicts that you are likely to have diabetes.")
    else:
        st.success("The model predicts that you are unlikely to have diabetes.")
    
