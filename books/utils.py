# books/utils.py
import joblib
import numpy as np
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'fake_user_model.pkl')
model = joblib.load(MODEL_PATH)

def check_if_fake_user(name, email, address, age, score, visit_count):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()

    # For real use, encode with training encoders
    features = np.array([[0, age, 0, score, visit_count, 0]])  # dummy encoded
    return model.predict(features)[0] == 1


# Load the model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "spam_model.pkl")
spam_model = joblib.load(MODEL_PATH)

# Function to predict spam
def is_spam_review(text):
    return bool(spam_model.predict([text])[0])  # 1 = spam, 0 = not spam

