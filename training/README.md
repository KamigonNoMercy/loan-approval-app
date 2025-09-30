# 🔧 Training Guide

This folder contains the scripts and resources to **train, evaluate, and test** the loan approval prediction model.

---

## 📂 Files
- `train_model.py` — script to train and export the model (`best_model.pkl`).
- `predict_model.py` — example of programmatic inference using the trained model.
- `README.md` — this guide.

---

## 🚀 How to Retrain the Model
1. Make sure you have installed all dependencies from the root project:
   ```bash
   pip install -r ../requirements.txt
2. Prepare your dataset (CSV format). Example:
   Dataset_A_loan.csv
3. Run the training script:
   python train_model.py
4. After training completes, the best model will be saved as:
   ../app/best_model.pkl

## 📊 Evaluation
- The script prints metrics such as accuracy, precision, recall, and F1-score.
- You can modify the train_model.py to test with different ML algorithms or hyperparameters.

## 🧪 Inference Example
To test predictions programmatically without Streamlit:
python predict_model.py
This will load best_model.pkl and run inference on a small sample.

## 📝 Notes
- Default model: XGBoost Classifier
- Features: demographic & financial attributes from the loan dataset
- Target: predict whether a loan application should be approved or rejected
