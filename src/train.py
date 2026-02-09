# src/train.py
"""
Simple training script for the Iris dataset from CSV.

Usage:
    python src/train.py
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

# 1. Load the dataset
data_path = "data/data.csv"
df = pd.read_csv(data_path)

# 2. Split features and target
X = df.drop(columns=["species"])
y = df["species"]

# 3. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the model
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate the model
preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)
print(f"Model accuracy: {accuracy:.4f}")

# 6. Save the model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/iris.joblib")
print("Model saved to models/iris.joblib")

