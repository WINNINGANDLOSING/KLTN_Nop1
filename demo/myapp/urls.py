from django.urls import path
from .views import chatbot, clear_chat_history, homepage, contact,  success

urlpatterns = [
    path('', homepage, name='homepage'),  # Default path
    path('homepage/', homepage, name='homepage'),  # Named path for homepage
    path('chatbot/', chatbot, name='chatbot'),

    path('clear-chat-history/', clear_chat_history, name='clear_chat_history'),
    path('contact/', contact, name='contact'),
    path('success/', success, name='success')
]