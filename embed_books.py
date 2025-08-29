import os
import django

# Setup Django environment
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "smart_library.settings")
django.setup()

from books.models import Book
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings():
    for book in Book.objects.all():
        if not book.summary_embedding:
            print(f"Embedding for: {book.title}")
            embedding = model.encode(book.summary).tolist()
            book.summary_embedding = embedding
            book.save()

if __name__ == "__main__":
    generate_embeddings()
