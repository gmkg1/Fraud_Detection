#!/usr/bin/env python

import os
import glob
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Use a non-interactive backend for matplotlib to avoid GUI errors
import matplotlib
matplotlib.use('Agg')

from flask import Flask, render_template, request
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
# ===============================================================
# Conversion Script (for Internship Dataset)
#
# This script was used to convert the .pkl files provided by the
# company into .csv format. 
# 
# It is commented out for documentation purposes.

# import os
# import pickle
# import pandas as pd
#
# output_dir = "E:/Internship/Fraud Detection/formatteddataset"
# os.makedirs(output_dir, exist_ok=True)
#
# for j in range(4, 10):
#     for i in range(1, 31):
#         if i < 10:
#             pkl_path = f"E:/Internship/Fraud Detection/dataset/2018-0{j}-0{i}.pkl"
#             csv_path = os.path.join(output_dir, f"2018-0{j}-0{i}.csv")
#         else:
#             pkl_path = f"E:/Internship/Fraud Detection/dataset/2018-0{j}-{i}.pkl"
#             csv_path = os.path.join(output_dir, f"2018-0{j}-{i}.csv")
#         
#         if not os.path.exists(pkl_path):
#             print(f"File not found: {pkl_path}, skipping.")
#             continue
#
#         with open(pkl_path, "rb") as f:
#             data = pickle.load(f)
#             data.to_csv(csv_path, index=False)
#             print(f"Converted {pkl_path} -> {csv_path}")

# ===============================================================

app = Flask(__name__)

MODEL_PATH = "fraud_model_rf.pkl"
THRESHOLD = 0.2

# =========================
# Load or Train Model + Encoders
# =========================
def load_or_train_model():
    """Loads a pre-trained model or trains a new one if it doesn't exist."""
    all_files = glob.glob("formatteddataset/*.csv")
    if not all_files:
        raise FileNotFoundError("No CSV files found in 'formatteddataset/'. Please check your dataset path.")

    df_list = [pd.read_csv(file) for file in all_files]
    full_df = pd.concat(df_list, ignore_index=True)

    # Feature engineering
    full_df["TX_DATETIME"] = pd.to_datetime(full_df["TX_DATETIME"])
    full_df["TX_HOUR"] = full_df["TX_DATETIME"].dt.hour
    full_df["TX_DAY_OF_WEEK"] = full_df["TX_DATETIME"].dt.dayofweek
    start_date = full_df['TX_DATETIME'].min()
    full_df['TX_DAYS_SINCE_START'] = (full_df['TX_DATETIME'] - start_date).dt.days

    # Save encoder categories
    full_df["CUSTOMER_ID_ENC"] = full_df["CUSTOMER_ID"].astype("category").cat.codes
    full_df["TERMINAL_ID_ENC"] = full_df["TERMINAL_ID"].astype("category").cat.codes
    
    customer_encoder = full_df["CUSTOMER_ID"].astype("category").cat.categories
    terminal_encoder = full_df["TERMINAL_ID"].astype("category").cat.categories

    features = [
        "TX_AMOUNT", "TX_HOUR", "TX_DAY_OF_WEEK",
        "TX_DAYS_SINCE_START", "CUSTOMER_ID_ENC", "TERMINAL_ID_ENC"
    ]

    X = full_df[features]
    y = full_df["TX_FRAUD"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print("Training data shape before SMOTE:", X_train.shape, y_train.shape)

    if os.path.exists(MODEL_PATH):
        print("Model found. Loading model bundle...")
        bundle = joblib.load(MODEL_PATH)
        model = bundle["model"]
        customer_encoder = bundle["customer_encoder"]
        terminal_encoder = bundle["terminal_encoder"]
        start_date = bundle["start_date"]
    else:
        print("Model not found. Applying SMOTE and training new model...")
        
        # Apply SMOTE to the training data only
        sm = SMOTE(random_state=42)
        X_resampled, y_resampled = sm.fit_resample(X_train, y_train)

        print("Training data shape after SMOTE:", X_resampled.shape, y_resampled.shape)

        rf_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            # class_weight is no longer needed after SMOTE
            n_jobs=-1
        )
        rf_model.fit(X_resampled, y_resampled)

        bundle = {
            "model": rf_model,
            "customer_encoder": customer_encoder,
            "terminal_encoder": terminal_encoder,
            "start_date": start_date
        }
        joblib.dump(bundle, MODEL_PATH)
        model = rf_model
    
    return model, X_test, y_test, features, customer_encoder, terminal_encoder, start_date

# Load model + encoders
try:
    model, X_test, y_test, FEATURES, customer_encoder, terminal_encoder, START_DATE = load_or_train_model()
    print("Model loaded successfully.")
except FileNotFoundError as e:
    print(e)
    exit()
except Exception as e:
    print(f"An unexpected error occurred during model loading: {e}")
    exit()

# =========================
# Routes
# =========================
@app.route("/")
def index():
    input_fields = ["TX_DATETIME", "CUSTOMER_ID", "TERMINAL_ID",
                    "TX_AMOUNT", "TX_TIME_SECONDS", "TX_TIME_DAYS"]
    return render_template("index.html", fields=input_fields)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect raw inputs
        TX_DATETIME = request.form["TX_DATETIME"]
        CUSTOMER_ID = request.form["CUSTOMER_ID"]
        TERMINAL_ID = request.form["TERMINAL_ID"]
        TX_AMOUNT = float(request.form["TX_AMOUNT"])

        # Convert datetime
        tx_datetime = pd.to_datetime(TX_DATETIME)
        TX_HOUR = tx_datetime.hour
        TX_DAY_OF_WEEK = tx_datetime.dayofweek
        TX_DAYS_SINCE_START = (tx_datetime - START_DATE).days

        # Encode using saved encoders, handling new categories gracefully
        try:
            cust_enc = np.where(customer_encoder == CUSTOMER_ID)[0][0]
        except IndexError:
            cust_enc = -1 

        try:
            term_enc = np.where(terminal_encoder == TERMINAL_ID)[0][0]
        except IndexError:
            term_enc = -1

        # Final feature vector
        features_dict = {
            "TX_AMOUNT": TX_AMOUNT,
            "TX_HOUR": TX_HOUR,
            "TX_DAY_OF_WEEK": TX_DAY_OF_WEEK,
            "TX_DAYS_SINCE_START": TX_DAYS_SINCE_START,
            "CUSTOMER_ID_ENC": cust_enc,
            "TERMINAL_ID_ENC": term_enc
        }
        df = pd.DataFrame([features_dict])

        # Prediction
        y_proba = model.predict_proba(df)[:, 1][0]
        prediction = int(y_proba >= THRESHOLD)

        return render_template(
            "result.html",
            proba=round(y_proba, 4),
            prediction=prediction,
            is_fraud=bool(prediction) 
        )

    except Exception as e:
        return f"Error during prediction: {e}"


@app.route("/confusion")
def confusion_view():
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred_adjusted = (y_proba >= THRESHOLD).astype(int)

        # Print detailed classification report to the console
        print("\n--- Classification Report ---")
        print(classification_report(y_test, y_pred_adjusted))

        # Calculate overall accuracy
        accuracy = accuracy_score(y_test, y_pred_adjusted)
        print(f"Overall Accuracy: {accuracy:.4f}")

        cm = confusion_matrix(y_test, y_pred_adjusted)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Not Fraud", "Fraud"],
                    yticklabels=["Not Fraud", "Fraud"])
        plt.title(f"Confusion Matrix (Threshold={THRESHOLD})")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")

        plot_path = "static/confusion.png"
        os.makedirs("static", exist_ok=True)
        plt.savefig(plot_path)
        plt.close()

        # You can also pass the report data to the template if you want to display it
        report = classification_report(y_test, y_pred_adjusted, output_dict=True)

        return render_template("confusion.html", plot_url=plot_path, report=report)

    except Exception as e:
        return f"Error generating confusion matrix: {e}"

if __name__ == "__main__":
    app.run(debug=True)