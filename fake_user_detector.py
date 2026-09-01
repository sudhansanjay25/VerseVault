import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Load dataset
df = pd.read_csv("fake_user_dataset.csv")

# Encode categorical features
le = LabelEncoder()
df['name_encoded'] = le.fit_transform(df['name'])
df['email_encoded'] = le.fit_transform(df['email'])
df['address_encoded'] = le.fit_transform(df['address'])
df['role_encoded'] = le.fit_transform(df['role'])  # 👈 new line

# Features & target
X = df[['name_encoded', 'age', 'email_encoded', 'score', 'visit_count', 'address_encoded', 'role_encoded']]
y = df['is_fake']

# Split & train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluation
print("✅ Model Trained")
print(classification_report(y_test, model.predict(X_test)))

# Save model
joblib.dump(model, "fake_user_model.pkl")
print("📦 Model saved as fake_user_model.pkl")
