import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv("fake_user_dataset.csv")

# Encode categorical variables
le = LabelEncoder()
df['name_encoded'] = le.fit_transform(df['name'])
df['email_encoded'] = le.fit_transform(df['email'])
df['address_encoded'] = le.fit_transform(df['address'])

# Features and target
X = df[['name_encoded', 'age', 'email_encoded', 'score', 'visit_count', 'address_encoded']]
y = df['is_fake']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("📊 Classification Report:")
print(classification_report(y_test, y_pred))
print("🔁 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(model, "fake_user_model.pkl")
print("✅ Model saved as fake_user_model.pkl")
