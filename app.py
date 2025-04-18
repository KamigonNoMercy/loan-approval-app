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
        model_bundle = pickle.load(f)
    return model_bundle

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
        # Tambahkan field manual loan_percent_income agar bisa dikoreksi
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
    edu_bachelor = st.checkbox("Bachelor")
    edu_master = st.checkbox("Master")
    edu_doctorate = st.checkbox("Doctorate")
    edu_highschool = st.checkbox("High School")

    st.markdown("### Status Tempat Tinggal")
    own_home = st.checkbox("OWN")
    rent_home = st.checkbox("RENT")
    other_home = st.checkbox("OTHER")

    st.markdown("### Tujuan Pinjaman")
    loan_education = st.checkbox("EDUCATION")
    loan_home = st.checkbox("HOMEIMPROVEMENT")
    loan_medical = st.checkbox("MEDICAL")
    loan_personal = st.checkbox("PERSONAL")
    loan_venture = st.checkbox("VENTURE")

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
        'person_education_Bachelor': edu_bachelor,
        'person_education_Doctorate': edu_doctorate,
        'person_education_High School': edu_highschool,
        'person_education_Master': edu_master,
        'person_home_ownership_OTHER': other_home,
        'person_home_ownership_OWN': own_home,
        'person_home_ownership_RENT': rent_home,
        'loan_intent_EDUCATION': loan_education,
        'loan_intent_HOMEIMPROVEMENT': loan_home,
        'loan_intent_MEDICAL': loan_medical,
        'loan_intent_PERSONAL': loan_personal,
        'loan_intent_VENTURE': loan_venture
    }])

    # Tambahkan kolom yang hilang (jaga-jaga)
    missing_cols = set(features) - set(df_input.columns)
    for col in missing_cols:
        df_input[col] = 0
    df_input = df_input[features]

    # Scaling hanya kolom yang perlu
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
