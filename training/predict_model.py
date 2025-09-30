import pickle
import pandas as pd

# Class untuk memuat model dan memprediksi
class LoanPredictor:
    # Konstruktor
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.scale_cols = None
        self.feature_names = None

    # Memuat file model pkl
    def load_model(self):
        with open(self.model_path, "rb") as f:
            model_package = pickle.load(f) # Load isi file pickle
            self.model = model_package['model'] # Ambil model
            self.scaler = model_package['scaler'] # Ambil scaler
            self.scale_cols = model_package['scale_cols']  # Ambil kolom scale
            self.feature_names = model_package['features']  # Ambil urutan fitur
        print("Model and metadata loaded.") 

    #  Melakukan preprocessing pada data input
    def preprocess(self, input_df):
        input_df = input_df.copy() # Salin data input
        input_df[self.scale_cols] = self.scaler.transform(input_df[self.scale_cols]) # Scale kolom numerik
        input_df = input_df.reindex(columns=self.feature_names) # Susun ulang urutan kolom

        # Cek jika ada NaN setelah reindex
        if input_df.isnull().any().any():
            missing = input_df.columns[input_df.isnull().any()].tolist()
            raise ValueError(f"⚠️ Kolom kosong terdeteksi: {missing}")
        return input_df
        
    # Melakukan prediksi terhadap data baru
    def predict(self, input_df):
        processed = self.preprocess(input_df) # Preprocess input
        pred = self.model.predict(processed)[0]  # Prediksi kelas
        prob = self.model.predict_proba(processed)[0][1] # Probabilitas disetujui
        return pred, prob # Return hasil

# Load predictor
predictor = LoanPredictor("best_model.pkl")
predictor.load_model()

# Ambil urutan kolom
features = predictor.feature_names

# Test Case 1
case_1 = pd.DataFrame([{
    'person_age': 22,
    'person_income_win': 71948,
    'person_emp_exp': 0,
    'loan_amnt': 35000,
    'loan_int_rate': 16.02,
    'loan_percent_income': 0.49,
    'cb_person_cred_hist_length': 3,
    'credit_score': 561,
    'person_gender': 0,
    'person_income': 71948,
    'previous_loan_defaults_on_file': 0,
    'person_education_Bachelor': 0,
    'person_education_Doctorate': 0,
    'person_education_High School': 0,
    'person_education_Master': 1,
    'person_home_ownership_OTHER': 0,
    'person_home_ownership_OWN': 0,
    'person_home_ownership_RENT': 1,
    'loan_intent_EDUCATION': 0,
    'loan_intent_HOMEIMPROVEMENT': 0,
    'loan_intent_MEDICAL': 0,
    'loan_intent_PERSONAL': 1,
    'loan_intent_VENTURE': 0
}], columns=features).astype(float)

pred1, prob1 = predictor.predict(case_1) # Prediksi test case 1
print("\n Test Case 1")
print("Hasil:", " Disetujui" if pred1 == 1 else " Ditolak")
print("Probabilitas disetujui: {:.2%}".format(prob1))

# Test Case 2
case_2 = pd.DataFrame([{
    'person_age': 21,
    'person_income_win': 12282,
    'person_emp_exp': 0,
    'loan_amnt': 1000,
    'loan_int_rate': 11.14,
    'loan_percent_income': 0.08,
    'cb_person_cred_hist_length': 2,
    'credit_score': 504,
    'person_gender': 0,
    'person_income': 12282,
    'previous_loan_defaults_on_file': 1,
    'person_education_Bachelor': 0,
    'person_education_Doctorate': 0,
    'person_education_High School': 1,
    'person_education_Master': 0,
    'person_home_ownership_OTHER': 0,
    'person_home_ownership_OWN': 1,
    'person_home_ownership_RENT': 0,
    'loan_intent_EDUCATION': 1,
    'loan_intent_HOMEIMPROVEMENT': 0,
    'loan_intent_MEDICAL': 0,
    'loan_intent_PERSONAL': 0,
    'loan_intent_VENTURE': 0
}], columns=features).astype(float)

pred2, prob2 = predictor.predict(case_2) # Prediksi test case 2
print("\n Test Case 2")
print("Hasil:", " Disetujui" if pred2 == 1 else " Ditolak")
print("Probabilitas disetujui: {:.2%}".format(prob2))