import streamlit as st
import joblib
import pandas as pd

model = joblib.load("model_hipertensi_rf.pkl")

st.title("Prediksi Risiko Hipertensi (Random Forest)")

age = st.number_input("Age", 1, 120, 30)
cigs = st.number_input("Cigarettes per Day", 0, 100, 0)
totchol = st.number_input("Total Cholesterol", 100, 600, 200)
sysbp = st.number_input("Systolic BP", 80, 250, 120)
diabp = st.number_input("Diastolic BP", 50, 150, 80)
bmi = st.number_input("BMI", 10.0, 60.0, 22.0)
heartrate = st.number_input("Heart Rate", 40, 200, 70)
glucose = st.number_input("Glucose", 40, 400, 80)

if st.button("Prediksi"):
    data = pd.DataFrame([{
        "age": age,
        "cigsPerDay": cigs,
        "totChol": totchol,
        "sysBP": sysbp,
        "diaBP": diabp,
        "BMI": bmi,
        "heartRate": heartrate,
        "glucose": glucose
    }])

    pred = model.predict(data)[0]
    prob = model.predict_proba(data)[0][1]

    st.subheader(f"Hasil Prediksi: {'RISIKO HIPERTENSI' if pred==1 else 'TIDAK BERISIKO'}")
    st.write(f"Probabilitas Risiko: {prob:.2f}")