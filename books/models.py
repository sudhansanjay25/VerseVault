from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
# Adjust if your imports differ
import os
import joblib
from textblob import TextBlob
from django.core.mail import send_mail

spam_model_path = os.path.join(os.path.dirname(__file__), 'spam_model.pkl')
spam_model = joblib.load(spam_model_path)


spam_model = joblib.load(spam_model_path)

class Visitor(models.Model):
    ROLE_CHOICES = [
        ('admin', 'Admin'),
        ('user', 'User'),
    ]

    name = models.CharField(max_length=100)
    age = models.IntegerField()
    mobile = models.CharField(max_length=15, unique=True)
    email = models.EmailField(unique=True)
    address = models.TextField()
    password = models.CharField(max_length=255)
    role = models.CharField(max_length=10, choices=ROLE_CHOICES)
    visit_count = models.IntegerField(default=0)
    score = models.IntegerField(default=0)
    created_at = models.DateTimeField(default=timezone.now)
    face_encoding = models.JSONField(null=True, blank=True)


    def __str__(self):
        return self.name


class Book(models.Model):
    title = models.CharField(max_length=200)
    author = models.CharField(max_length=100)
    publisher = models.CharField(max_length=100)
    publication_year = models.IntegerField()
    summary = models.TextField()
    available_copies = models.IntegerField(default=1)
    borrow_count = models.IntegerField(default=0)
    summary_embedding = models.JSONField(null=True, blank=True)

    rack_number = models.CharField(max_length=10, blank=True, null=True)
    shelf_number = models.CharField(max_length=10, blank=True, null=True)

    def __str__(self):
        return self.title


class BookLocation(models.Model):
    book = models.OneToOneField(Book, on_delete=models.CASCADE)
    block = models.CharField(max_length=20)
    row = models.CharField(max_length=10)
    column = models.CharField(max_length=10)
    position = models.CharField(max_length=10)

    def __str__(self):
        return f"{self.book.title} - Block {self.block}, Row {self.row}, Column {self.column}, Pos {self.position}"


class BorrowRecord(models.Model):
    visitor = models.ForeignKey(Visitor, on_delete=models.CASCADE, related_name='borrow_records')
    book = models.ForeignKey(Book, on_delete=models.CASCADE)
    borrow_date = models.DateField(auto_now_add=True)
    return_date = models.DateField(null=True, blank=True)
    due_date = models.DateField(null=True, blank=True)
    is_returned = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.visitor.name} borrowed {self.book.title}"





# Load your spam detection model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'spam_model.pkl')
spam_model = joblib.load(MODEL_PATH)

class BookReview(models.Model):
    

    visitor = models.ForeignKey(Visitor, on_delete=models.CASCADE)
    book = models.ForeignKey(Book, on_delete=models.CASCADE)
    review = models.TextField()
    is_spam = models.BooleanField(default=False)
    sentiment = models.CharField(max_length=10, default="Neutral")
    created_at = models.DateTimeField(default=timezone.now)

    def save(self, *args, **kwargs):
        # Spam detection
        prediction = spam_model.predict([self.review])[0]
        if self.is_spam:
            self.delete()
            return  # Skip saving


        if self.is_spam:
            send_mail(
                subject="⚠️ Inappropriate Review Detected",
                message=f"Hi {self.visitor.name},\n\nYour recent review on '{self.book.title}' was flagged as spam. Please ensure your reviews are meaningful and appropriate.\n\nThanks,\nSmart Library Admin",
                from_email=None,
                recipient_list=[self.visitor.email],
                fail_silently=False,
            )

        # Sentiment analysis
        blob = TextBlob(self.review)
        polarity = blob.sentiment.polarity
        if polarity > 0.2:
            self.sentiment = "Positive"
        elif polarity < -0.2:
            self.sentiment = "Negative"
        else:
            self.sentiment = "Neutral"

        super().save(*args, **kwargs)





    def __str__(self):
        return f"{self.visitor.name}'s review on {self.book.title}"


    def __str__(self):
        return f"{self.visitor.name}'s review on {self.book.title}"




class AdminProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    profile_picture = models.ImageField(upload_to="admin_profiles/", null=True, blank=True)

    def __str__(self):
        return self.user.username


# books/models.py
class VisitorLog(models.Model):
    name = models.CharField(max_length=100)
    face_encoding = models.JSONField()
    timestamp = models.DateTimeField(default=timezone.now)
    status = models.CharField(max_length=20, choices=[('Authorized', 'Authorized'), ('Unauthorized', 'Unauthorized')])

    def __str__(self):
        return f"{self.name} - {self.status} - {self.timestamp.strftime('%Y-%m-%d %H:%M')}"
