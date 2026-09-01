📚 Smart Library Management System 🚀

A full-stack AI-powered smart library system built with Django that automates library management with ML/NLP features like spam review detection, sentiment analysis, voice-based book search, personalized recommendations, and automatic email alerts. Designed for both administrators and students to offer a seamless digital library experience.



🔥 Key Features

✅ Authentication & Role-Based Access

Visitor registration/login with password encryption.

Admin dashboard vs. User dashboard with dynamic content.

Session-based role detection and redirection logic.



📖 Book Management

Add, update, and assign books with custom rack and shelf info.

Borrow and return functionality with real-time updates on available copies.

Borrow history with status (on-time, late, returned).




✨ Personalized Recommendations

Book recommendation engine using Sentence Transformers (all-MiniLM-L6-v2) for vector-based similarity.

Learns from each visitor’s history and gives AI-powered suggestions.




🧠 Spam Review Detection (ML)

Trained a model using scikit-learn on spam review datasets.

Reviews are automatically flagged as spam/non-spam using a pickled spam_model.pkl and vectorizer.pkl.

Spam reviews trigger an auto-delete or alert email (configurable).





📊 Sentiment Analysis (NLP)

Integrated TextBlob to evaluate if reviews are Positive, Negative, or Neutral.

Feedback is displayed with visual emojis and admin filtering.





🔍 Voice-Based Book Search

Offline voice recognition using VOSK.

Converts speech to text and matches it with the book database using sentence embeddings.





📈 Book Popularity Prediction (ML)

Predicts popular books using historical borrow counts.

Uses Linear Regression (can be extended to XGBoost or Random Forest).





📬 Automated Email Notifications

✅ Email confirmation on book borrow and return.

⏰ Due date reminder with fine info.

🚨 Spam review warning to users.

Emails are powered by Django’s send_mail() function (configurable with SMTP).





🛠️ Tech Stack

| Layer           | Technology                                 |
| --------------- | ------------------------------------------ |
| Backend         | Django (Python)                            |
| Frontend        | HTML + Tailwind CSS                        |
| ML/NLP Models   | scikit-learn, TextBlob, VOSK, Transformers |
| Embedding Model | SentenceTransformer (MiniLM-L6-v2)         |
| Database        | SQLite (default)                           |
| Email Service   | Django email backend (SMTP-based)          |
| Speech to Text  | VOSK Offline Model                         |





💡 Future Scope

📸 OCR for book auto-fill (title/author from cover).

🧾 Summary generation using LLMs.

🌐 Cloud-based face login with live camera detection.

🎯 Real-time fake user detection improvements.

📱 PWA app version for mobile access.



⚙️ Installation & Setup

git clone https:https://github.com/sajith71/smart-library.git

cd smart-library

pip install -r requirements.txt

python manage.py migrate





python manage.py runserver




