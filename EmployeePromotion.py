import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Load and preprocess dataset
def load_and_preprocess_data(filepath="employee_promotion_dataset.csv"):
    print("📥 Loading and preprocessing dataset...")
    df = pd.read_csv(filepath)
    df.fillna(0, inplace=True)
    df = pd.get_dummies(df)  # handle categorical variables

    X = df.drop("promotion", axis=1)
    y = df["promotion"]
    print(f"✅ Loaded {len(df)} records with {X.shape[1]} features.\n")
    return df, X, y

# 2. Perform EDA
def perform_eda(df):
    count_above_35 = df[df["age"] > 35].shape[0]
    print(f"📊 Number of employees with age > 35: {count_above_35}")
    return count_above_35

# 3. Split the data
def split_data(X, y, test_size=0.2):
    print("✂️ Splitting data...")
    return train_test_split(X, y, test_size=test_size, random_state=42)

# 4. Train and save model
def train_and_save_model(X_train, y_train, model_path="promotion_svm_model.pkl"):
    print("🏋️ Training SVM model...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = SVC(kernel='linear', probability=True)
    model.fit(X_train_scaled, y_train)

    joblib.dump(model, model_path)
    print(f"✅ Model saved to '{model_path}'\n")
    return model, scaler

# 5. Predict from JSON input
def predict_from_json(model, scaler, json_file="new_employee.json", feature_columns=None):
    print(f"📄 Reading input from '{json_file}'...")
    try:
        with open(json_file, 'r') as file:
            input_data = json.load(file)

        df_input = pd.DataFrame([input_data["employee"]])
        df_input = pd.get_dummies(df_input)

        # Ensure input matches feature columns
        for col in feature_columns:
            if col not in df_input.columns:
                df_input[col] = 0
        df_input = df_input[feature_columns]

        scaled_input = scaler.transform(df_input)
        prediction = model.predict(scaled_input)[0]

        print(f"🔮 Prediction: {'Promoted' if prediction == 1 else 'Not Promoted'}")
        return prediction
    except Exception as e:
        print(f"❌ Error during prediction: {e}")

# ========== Pipeline Execution ==========
if __name__ == "__main__":
    df, X, y = load_and_preprocess_data("employee_promotion_dataset.csv")
    perform_eda(df)  # Count employees older than 35
    X_train, X_test, y_train, y_test = split_data(X, y)
    model, scaler = train_and_save_model(X_train, y_train)
    feature_columns = X.columns.tolist()
    predict_from_json(model, scaler, json_file="new_employee.json", feature_columns=feature_columns)
