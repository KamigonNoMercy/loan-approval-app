import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Loan Approval Predictor", layout="centered")
st.title("🏦 Loan Approval Prediction App")
st.write("Masukkan data peminjam untuk memprediksi apakah pinjaman akan disetujui atau tidak.")

# Load model
@st.cache_resource
def load_model():
    with open("best_model.pkl", "rb") as f:
        return pickle.load(f)

model_bundle = load_model()
model = model_bundle['model']
scaler = model_bundle['scaler']
scale_cols = model_bundle['scale_cols']
features = model_bundle['features']

# Input form
with st.form("loan_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        person_age = st.number_input("Usia Pemohon", 18, 100, 30)
        person_income = st.number_input("Pendapatan Tahunan", 1000, 1_000_000, 50000)
        person_emp_exp = st.slider("Lama Pengalaman Kerja (tahun)", 0, 40, 5)
        loan_amnt = st.number_input("Jumlah Pinjaman", 500, 50000, 10000)
        loan_int_rate = st.number_input("Suku Bunga (%)", 5.0, 30.0, 12.5)

    with col2:
        default_percent = round(loan_amnt / max(person_income, 1), 2)
        loan_percent_income = st.number_input(
            "Proporsi Pinjaman terhadap Pendapatan",
            min_value=0.00, max_value=2.00, value=default_percent, step=0.01
        )
        cred_hist_length = st.slider("Lama Histori Kredit (tahun)", 0, 30, 5)
        credit_score = st.slider("Skor Kredit", 300, 850, 650)
        person_gender = st.radio("Jenis Kelamin", ["Laki-laki", "Perempuan"])
        prev_default = st.radio("Pernah Gagal Bayar?", ["Ya", "Tidak"])

    st.markdown("### Kategori Pendidikan")
    edu_level = st.selectbox("Pilih Salah Satu", ["Bachelor", "Master", "Doctorate", "High School"])

    st.markdown("### Status Tempat Tinggal")
    home_status = st.selectbox("Pilih Salah Satu", ["OWN", "RENT", "OTHER"])

    st.markdown("### Tujuan Pinjaman")
    loan_intent = st.selectbox("Pilih Salah Satu", ["EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"])

    submitted = st.form_submit_button("Prediksi")

# Prediksi
if submitted:
    df_input = pd.DataFrame([{
        'person_age': person_age,
        'person_income_win': person_income,
        'person_emp_exp': person_emp_exp,
        'loan_amnt': loan_amnt,
        'loan_int_rate': loan_int_rate,
        'loan_percent_income': loan_percent_income,
        'cb_person_cred_hist_length': cred_hist_length,
        'credit_score': credit_score,
        'person_gender': 1 if person_gender == "Laki-laki" else 0,
        'previous_loan_defaults_on_file': 1 if prev_default == "Ya" else 0,

        # One-hot encode single education level
        'person_education_Bachelor': 1 if edu_level == "Bachelor" else 0,
        'person_education_Master': 1 if edu_level == "Master" else 0,
        'person_education_Doctorate': 1 if edu_level == "Doctorate" else 0,
        'person_education_High School': 1 if edu_level == "High School" else 0,

        # One-hot encode home status
        'person_home_ownership_OWN': 1 if home_status == "OWN" else 0,
        'person_home_ownership_RENT': 1 if home_status == "RENT" else 0,
        'person_home_ownership_OTHER': 1 if home_status == "OTHER" else 0,

        # One-hot encode loan intent
        'loan_intent_EDUCATION': 1 if loan_intent == "EDUCATION" else 0,
        'loan_intent_HOMEIMPROVEMENT': 1 if loan_intent == "HOMEIMPROVEMENT" else 0,
        'loan_intent_MEDICAL': 1 if loan_intent == "MEDICAL" else 0,
        'loan_intent_PERSONAL': 1 if loan_intent == "PERSONAL" else 0,
        'loan_intent_VENTURE': 1 if loan_intent == "VENTURE" else 0
    }])

    # Tambahkan kolom hilang jika ada
    for col in (set(features) - set(df_input.columns)):
        df_input[col] = 0
    df_input = df_input[features]

    # Scaling
    df_scaled = pd.concat([
        pd.DataFrame(scaler.transform(df_input[scale_cols]), columns=scale_cols),
        df_input.drop(columns=scale_cols).reset_index(drop=True)
    ], axis=1)

    # Prediksi
    try:
        prediction = model.predict(df_scaled)[0]
        probability = float(model.predict_proba(df_scaled)[0][1])
        label = "Pinjaman Disetujui" if prediction == 1 else "Pinjaman Ditolak"
        emoji = "✅" if prediction == 1 else "❌"

        st.subheader("Hasil Prediksi:")
        if prediction == 1:
            st.success(f"{emoji} {label} (Probabilitas: {probability:.2%})")
        else:
            st.error(f"{emoji} {label} (Probabilitas: {probability:.2%})")
    except Exception as e:
        st.error("Terjadi kesalahan saat menampilkan hasil.")
        st.code(str(e))
