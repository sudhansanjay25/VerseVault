# train_vectorizer.py
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import os

# 🚨 Option A: Use your own BookReview model if you have data
try:
    from books.models import BookReview
    from django.conf import settings
    import django
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'smart_library.settings')
    django.setup()
    reviews = BookReview.objects.values_list('review', flat=True)
except:
    # 🚨 Option B: Fallback if BookReview model fails (dummy reviews)
    reviews = [
        "This book is amazing",
        "I really enjoyed this book",
        "Terrible experience, not recommended",
        "Decent read but not very engaging",
        "Absolutely loved it, 5 stars!"
    ]

# 🧪 Train the vectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit(reviews)

# 💾 Save vectorizer
output_path = os.path.join(os.path.dirname(__file__), 'vectorizer.pkl')
joblib.dump(vectorizer, output_path)

print(f"✅ Vectorizer saved to {output_path}")
