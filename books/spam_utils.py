import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report


def train_spam_detector(csv_path="spam_reviews.csv", output_path="books"):
    # Load dataset
    df = pd.read_csv(csv_path)

    # Check required columns
    if 'review_text' not in df.columns or 'is_spam' not in df.columns:
        raise ValueError("CSV must contain 'review_text' and 'is_spam' columns.")

    X = df['review_text']
    y = df['is_spam']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build model pipeline
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(X_train, y_train)

    # Evaluate the model
    print("📊 Classification Report:\n")
    print(classification_report(y_test, model.predict(X_test)))

    # Save the model
        # Save the model
    os.makedirs(output_path, exist_ok=True)  # 🛠️ This line ensures the folder exists
    model_path = os.path.join(output_path, "spam_model.pkl")
    joblib.dump(model, model_path)

    print("✅ Spam detection model saved at:", model_path)


if __name__ == "__main__":
    # By default trains from spam_reviews.csv and saves to /books
    train_spam_detector()
