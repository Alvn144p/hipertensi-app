st.divider()
st.subheader("Prediksi Massal (Upload CSV)")

uploaded_file = st.file_uploader("Upload CSV (schema harus sama dengan data training)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("Preview data:", df.head())

    try:
        preds = model.predict(df)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(df)[:, 1]
            df["prediksi_risiko"] = preds
            df["prob_risiko"] = probs
        else:
            df["prediksi_risiko"] = preds

        st.success("Prediksi berhasil.")
        st.dataframe(df.head(20))

        csv_out = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download hasil prediksi (CSV)",
            data=csv_out,
            file_name="hasil_prediksi.csv",
            mime="text/csv",
        )
    except Exception as e:
        st.error(f"Schema CSV tidak cocok dengan model: {e}")