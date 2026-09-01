from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import Book
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

@receiver(post_save, sender=Book)
def generate_summary_embedding(sender, instance, created, **kwargs):
    if created and not instance.summary_embedding:
        embedding = model.encode(instance.summary).tolist()
        instance.summary_embedding = embedding
        instance.save()
