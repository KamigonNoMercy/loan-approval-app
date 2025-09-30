import pandas as pd
import numpy as np
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats.mstats import winsorize
from sklearn.metrics import (
    classification_report, accuracy_score,
    precision_score, recall_score, f1_score
)
import warnings
warnings.filterwarnings("ignore")

# CLASS: LoanXGBTrainer
class LoanXGBTrainer:
    # Konstruktor
    def __init__(self, data_path):
        # Inisialisasi atribut penting
        self.data_path = data_path          # Path file dataset
        self.df = None                      # Dataset asli
        self.model = None                   # Model XGBoost
        self.scaler = None                  # Scaler untuk kolom numerik
        self.features = None                # Kolom akhir yang digunakan model
        self.scale_cols = [                 # Kolom numerik yang akan di-scale
            'person_age', 'person_income_win', 'person_emp_exp',
            'loan_amnt', 'loan_int_rate', 'loan_percent_income',
            'cb_person_cred_hist_length', 'credit_score'
        ]

    # Load & Clean
    def load_and_clean_data(self):
        df = pd.read_csv(self.data_path)

        # Median imputing
        df['person_income'] = df['person_income'].fillna(df['person_income'].median())
        
        df = df[df['person_age'] <= 100]

        # Benarkan format 
        df['person_gender'] = df['person_gender'].str.lower().str.strip()
        gender_map = {
            'fe male': 'female', 'femail': 'female', 'femle': 'female',
            'female': 'female', 'male': 'male', 'm': 'male'
        }
        df['person_gender'] = df['person_gender'].map(gender_map).fillna(df['person_gender'])

        # Winsorize income (batas atas 1%)
        df['person_income_win'] = winsorize(df['person_income'], limits=[0, 0.01])

        # Label encoding untuk kolom biner
        df['person_gender'] = df['person_gender'].map({'female': 0, 'male': 1})
        df['previous_loan_defaults_on_file'] = df['previous_loan_defaults_on_file'].map({'No': 0, 'Yes': 1})

        # One-hot encoding untuk kolom kategorikal multikategori
        df = pd.get_dummies(df, columns=['person_education', 'person_home_ownership', 'loan_intent'], drop_first=True)

        self.df = df
        print(" Data loaded and cleaned.")
        
    # Preprocessing
    def prepare_data(self):
        df = self.df.copy()
        X = df.drop(columns='loan_status')
        y = df['loan_status']

        # Split data menjadi training dan testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scaling hanya untuk kolom numerik
        self.scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train[self.scale_cols]),
            columns=self.scale_cols, index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test[self.scale_cols]),
            columns=self.scale_cols, index=X_test.index
        )

        # Gabungkan kolom yang sudah di-scale dengan kolom non-numerik
        X_train_final = pd.concat([X_train_scaled, X_train.drop(columns=self.scale_cols)], axis=1)
        X_test_final = pd.concat([X_test_scaled, X_test.drop(columns=self.scale_cols)], axis=1)

        # Simpan hasil akhir dan urutan kolom yang digunakan model
        self.X_train = X_train_final
        self.X_test = X_test_final
        self.y_train = y_train
        self.y_test = y_test
        self.features = X_train_final.columns.tolist()  # Inilah yang digunakan saat inference
        print(" Data prepared and scaled.")

    # Train the Model
    def train_best_model(self):
        # Inisialisasi dan latih model XGBoost
        self.model = XGBClassifier(
            colsample_bytree=0.8,
            learning_rate=0.1,
            max_depth=6,
            n_estimators=200,
            subsample=0.8,
            eval_metric='logloss',
            random_state=42
        )
        self.model.fit(self.X_train, self.y_train)
        print("Model trained.")

    # Evaluate Model
    def evaluate(self):
        preds = self.model.predict(self.X_test)
        print("\n Evaluation Report:")
        print("Accuracy :", accuracy_score(self.y_test, preds))
        print("Precision:", precision_score(self.y_test, preds))
        print("Recall   :", recall_score(self.y_test, preds))
        print("F1-score :", f1_score(self.y_test, preds))
        print("\nClassification Report:\n", classification_report(self.y_test, preds))

    # Save Model
    def save_model(self, out_path="best_model.pkl"):
        with open(out_path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "scale_cols": self.scale_cols,
                "features": self.features
            }, f)
        print(f" Model saved to {out_path}")


# Main Guard
if __name__ == "__main__":
    trainer = LoanXGBTrainer("Dataset_A_loan.csv")
    trainer.load_and_clean_data()
    trainer.prepare_data()
    trainer.train_best_model()
    trainer.evaluate()
    trainer.save_model("best_model.pkl")

