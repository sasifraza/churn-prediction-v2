import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "churn.csv")
MODEL_PATH = os.path.join(BASE_DIR, "..", "saved_models", "model.pkl")

# Load data
def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f"Data loaded: {df.shape}")
    print(df.head())
    return df

# Preprocess data
def preprocess_data(df):
    df = df.copy()

    # Drop customer ID
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Fix TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.fillna(0)

    # Encode target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Encode categoricals
    df = pd.get_dummies(df, drop_first=True)

    print(f"Preprocessed: {df.shape}")
    return df

# Train model
def train_model(df):
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train RandomForest
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate at default threshold
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, model.predict_proba(X_test)[:,1]):.4f}")

    # Tune threshold to improve recall
    y_proba = model.predict_proba(X_test)[:,1]
    y_pred_tuned = (y_proba >= 0.3).astype(int)
    print(f"\nAfter threshold tuning (0.3):")
    print(f"Recall: {recall_score(y_test, y_pred_tuned):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred_tuned):.4f}")

    return model

# Save model
def save_model(model):
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

# Main
if __name__ == "__main__":
    df = load_data()
    df = preprocess_data(df)
    model = train_model(df)
    save_model(model)
    print("Training complete!")






