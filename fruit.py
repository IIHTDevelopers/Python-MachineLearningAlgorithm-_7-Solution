import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Load real-world dataset
def load_fruit_data():
    print("📥 Loading fruit dataset...")
    url = "fruit_data.csv"  # Your CSV file with fruit features
    df = pd.read_csv(url)
    print(f"✅ Loaded {len(df)} records.\n")
    return df

# 2. Preprocess data
def preprocess_fruit_data(df):
    print("🛠️ Preprocessing data...")
    X = df.drop("fruit_name", axis=1)
    y = df["fruit_name"]
    print("✅ Features and target separated.\n")
    return X, y

# 3. Split the data
def split_fruit_data(X, y, test_size=0.2):
    print("✂️ Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    print(f"✅ Train: {len(X_train)}, Test: {len(X_test)}\n")
    return X_train, X_test, y_train, y_test

# 4. Create model
def create_model(n_estimators=100, max_depth=None):
    print("🔧 Creating Random Forest model...")
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    return model

# 5. Train model
def train_model(model, X_train, y_train):
    print("🏋️ Training model...")
    model.fit(X_train, y_train)
    print("✅ Training complete.\n")
    return model

# 6. Save model
def save_model(model, filename="fruit_rf_model.pkl"):
    print(f"💾 Saving model as '{filename}'...")
    joblib.dump(model, filename)
    print("✅ Model saved.\n")

# 7. Load model
def load_model(filename="fruit_rf_model.pkl"):
    print(f"📦 Loading model from '{filename}'...")
    model = joblib.load(filename)
    print("✅ Model loaded.\n")
    return model

# 8. Predict new fruit from JSON
def check_new_data_from_json(model, json_file="fruit_item.json"):
    import json

    print(f"📄 Checking new fruit item from {json_file}...")
    try:
        with open(json_file, 'r') as file:
            data = json.load(file)

        item = data['fruit']

        item_df = pd.DataFrame([item])

        print(f"🔍 Making prediction on NEW FRUIT ITEM...")
        prediction = model.predict(item_df)[0]

        print("🍏 Fruit Input:")
        print(item_df.to_string(index=False))

        print(f"\n🔮 Predicted Fruit: {prediction}")

    except Exception as e:
        print(f"❌ Error checking new data: {e}\n")

# --- Pipeline Execution ---
df = load_fruit_data()
X, y = preprocess_fruit_data(df)
X_train, X_test, y_train, y_test = split_fruit_data(X, y)
model = create_model()
trained_model = train_model(model, X_train, y_train)
save_model(trained_model)

# Check new data from JSON
check_new_data_from_json(trained_model)
