# load_books.py
import csv
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'smart_library.settings')
django.setup()

from books.models import Book

with open('book_dataset_100.csv', newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        Book.objects.create(
            title=row['title'],
            author=row['author'],
            publisher=row['publisher'],
            publication_year=int(row['publication_year']),
            summary=row['summary'],
            available_copies=int(row['available_copies']),
            borrow_count=int(row['borrow_count']),
        )
print("✅ Book data loaded successfully.")
