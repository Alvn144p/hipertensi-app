import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Prediksi Risiko Hipertensi", layout="centered")

st.title("Prediksi Risiko Hipertensi (Random Forest)")

# Load model
model = joblib.load("rf_deploy.pkl")  # pastikan file ini ada di repo

# Input UI
male = st.selectbox("Jenis Kelamin", ["Perempuan", "Laki-laki"])
male = 1 if male == "Laki-laki" else 0

age = st.number_input("Age", 18, 100, 34)
currentSmoker = st.selectbox("Perokok?", ["Tidak", "Ya"])
currentSmoker = 1 if currentSmoker == "Ya" else 0

cigs = st.number_input("Cigarettes per Day", 0, 100, 0)
bpmeds = st.selectbox("Minum Obat BP?", ["Tidak", "Ya"])
bpmeds = 1 if bpmeds == "Ya" else 0

diabetes = st.selectbox("Diabetes?", ["Tidak", "Ya"])
diabetes = 1 if diabetes == "Ya" else 0

totchol = st.number_input("Total Cholesterol", 100, 400, 200)
sysbp = st.number_input("Systolic BP", 80, 250, 120)
diabp = st.number_input("Diastolic BP", 40, 150, 80)
bmi = st.number_input("BMI", 10.0, 60.0, 22.0)
heartrate = st.number_input("Heart Rate", 40, 150, 70)
glucose = st.number_input("Glucose", 40, 400, 80)

if st.button("Prediksi"):
    data = pd.DataFrame([{
        "male": male,
        "age": age,
        "currentSmoker": currentSmoker,
        "cigsPerDay": cigs,
        "BPMeds": bpmeds,
        "diabetes": diabetes,
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
    st.write(f"Probabilitas risiko: **{prob:.2%}**")