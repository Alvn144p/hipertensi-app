import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Prediksi Risiko Hipertensi", layout="centered")
st.title("Prediksi Risiko Hipertensi (RF & SVM + SMOTE)")

# Pilih model
model_choice = st.selectbox(
    "Pilih Model",
    ["Random Forest (Baseline)", "Random Forest (+SMOTE)", "SVM (Baseline)", "SVM (+SMOTE)"]
)

# Load model sesuai pilihan
if model_choice == "Random Forest (Baseline)":
    model = joblib.load("rf_baseline.pkl")
elif model_choice == "Random Forest (+SMOTE)":
    model = joblib.load("rf_smote.pkl")
elif model_choice == "SVM (Baseline)":
    model = joblib.load("svm_baseline.pkl")
else:
    model = joblib.load("svm_smote.pkl")

st.subheader("Input Data Pasien")

male = st.selectbox("Jenis Kelamin (Male=1, Female=0)", [0, 1])
age = st.number_input("Age", min_value=1, max_value=120, value=30)
cigs = st.number_input("Cigarettes per Day", min_value=0, max_value=100, value=0)
bpMeds = st.selectbox("BP Medication (No=0, Yes=1)", [0, 1])
totchol = st.number_input("Total Cholesterol", min_value=100, max_value=600, value=200)
sysbp = st.number_input("Systolic BP", min_value=80, max_value=250, value=120)
diabp = st.number_input("Diastolic BP", min_value=50, max_value=150, value=80)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=22.0)
heartrate = st.number_input("Heart Rate", min_value=40, max_value=200, value=70)
glucose = st.number_input("Glucose", min_value=40, max_value=400, value=80)

if st.button("Prediksi"):
    data = pd.DataFrame([{
        "male": male,
        "age": age,
        "cigsPerDay": cigs,
        "bpMeds": bpMeds,
        "totChol": totchol,
        "sysBP": sysbp,
        "diaBP": diabp,
        "BMI": bmi,
        "heartRate": heartrate,
        "glucose": glucose
    }])

    pred = model.predict(data)[0]

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(data)[0][1]
        st.success(f"Hasil: {'RISIKO HIPERTENSI' if pred==1 else 'TIDAK BERISIKO'}")
        st.write(f"Probabilitas risiko: {prob:.2f}")
    else:
        st.success(f"Hasil: {'RISIKO HIPERTENSI' if pred==1 else 'TIDAK BERISIKO'}")
        st.info("Model ini tidak menyediakan probabilitas.")