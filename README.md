# 🏦 Loan Approval Prediction (XGBoost + Streamlit)

End-to-end ML app to predict **loan approvals** using demographic and financial features.  
The model is trained with **XGBoost**, bundled with preprocessing (scaler & feature schema),  
and deployed as an interactive **Streamlit** app.

---

## ✨ Features
- Simple form-based UI for loan application input.
- Model output: **Approved / Rejected** + probability.
- Scaler + one-hot encoding schema embedded inside `best_model.pkl`.
- Works both locally and on **Streamlit Community Cloud**.

---

## 📦 Repository Layout
```
loan-approval-app/
├─ app/
│ ├─ app.py # Streamlit UI (uses model .pkl)
│ └─ best_model.pkl # trained model bundle
├─ training/
│ ├─ train_model.py # script to train & export model
│ ├─ predict_model.py # example of programmatic inference
│ └─ README.md # how to retrain & evaluate
├─ requirements.txt
├─ README.md # this file
├─ .gitignore
├─ LICENSE (MIT)
└─ .streamlit/
└─ config.toml # optional theme settings
```

---

## 🚀 Run Locally
```bash
git clone https://github.com/<your-username>/loan-approval-app.git
cd loan-approval-app
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/app.py
