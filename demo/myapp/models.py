# models.py

from django.db import models

class ChatMessage(models.Model):
    message = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
    origin = models.CharField(max_length=20)  # e.g., 'human' or 'bot'
