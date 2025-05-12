import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import json

# 1. Load and preprocess dataset
def load_and_preprocess_data(filepath="fruit_data.csv"):
    print("📥 Loading and preprocessing dataset...")
    df = pd.read_csv(filepath)
    X = df.drop("fruit_name", axis=1)
    y = df["fruit_name"]
    print(f"✅ Loaded {len(df)} records and separated features/target.\n")
    return X, y

# 2. Split the data
def split_data(X, y, test_size=0.2):
    print("✂️ Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    print(f"✅ Split complete: Train={len(X_train)}, Test={len(X_test)}\n")
    return X_train, X_test, y_train, y_test

# 3. Train and save model
def train_and_save_model(X_train, y_train, model_path="fruit_rf_model.pkl", n_estimators=100, max_depth=None):
    print("🏋️ Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    print(f"✅ Model trained and saved as '{model_path}'.\n")
    return model

# 4. Load trained model
def load_model(model_path="fruit_rf_model.pkl"):
    print(f"📦 Loading model from '{model_path}'...")
    model = joblib.load(model_path)
    print("✅ Model loaded.\n")
    return model

# 5. Predict from JSON input
def predict_from_json(model, json_file="fruit_item.json"):
    print(f"📄 Reading input from '{json_file}'...")
    try:
        with open(json_file, 'r') as file:
            data = json.load(file)
        item_df = pd.DataFrame([data['fruit']])
        print(f"🔍 Input Data:\n{item_df.to_string(index=False)}")
        prediction = model.predict(item_df)[0]
        print(f"🔮 Predicted Fruit: {prediction}")
        return prediction
    except Exception as e:
        print(f"❌ Error: {e}")

# Example pipeline execution
if __name__ == "__main__":
    X, y = load_and_preprocess_data("fruit_data.csv")
    X_train, X_test, y_train, y_test = split_data(X, y)
    trained_model = train_and_save_model(X_train, y_train)
    loaded_model = load_model()
    predict_from_json(loaded_model, "fruit_item.json")
