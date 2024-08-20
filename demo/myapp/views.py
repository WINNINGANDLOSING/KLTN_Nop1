# views.py

from django.shortcuts import render, redirect
from .models import ChatMessage
from .forms import QuestionForm
from llama_index.llms.ollama import Ollama

from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.deeplake import DeepLakeVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.schema import QueryBundle
from llama_index.llms.gemini import Gemini
from django.shortcuts import render
from django.core.mail import send_mail
from django.http import JsonResponse
from django.conf import settings
import json
import os

from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.postprocessor.cohere_rerank import CohereRerank

from llama_index.llms.ollama import Ollama
from llama_index.core.settings import Settings

from dotenv import load_dotenv
from django.conf import settings


from django.http import HttpResponse

from .forms import ContactForm
from django.core.mail import EmailMessage
from django.core.mail import send_mail
from .prompts import gen_qa_prompt, gen_rag_answer
from .RAG import RAG
from .RAG import format_chatbot_output

load_dotenv()
def give_answer(question):
    para = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
    return para

def contact(request):
    if request.method == 'POST':
        form = ContactForm(request.POST)
        if form.is_valid():
            name = form.cleaned_data['name']
            email = form.cleaned_data['email']
            message = form.cleaned_data['message']


            EmailMessage(
                'Contact Form Submission from {}'.format(name),
                message,
                email, # Send from (your website)
                ['khoalieudang@gmail.com'], # Send to (your admin email)
                [],
                reply_to=[email] # Email from the form to get back to
            ).send()

            return redirect('success')
    else:
        form = ContactForm()
    return render(request, 'contact.html', {'form': form})



def success(request):
    return render(request, 'success.html', {'message': 'Thank you for contacting us!'})    


def homepage(request):
    return render(request, 'homepage.html')


def chatbot(request):
    
    rag = RAG(
        model_name="vistral"    # vistral, phogpt, vinallama
    )
    # Retrieve all chat messages
    chat_history = ChatMessage.objects.all()
    
    # Create a form instance
    form = QuestionForm(request.POST or None)
    
    if request.method == 'POST':
        if form.is_valid():
            # Get the user's question from the form
            question = form.cleaned_data['question']
            
            # Save the user's question to the chat history
            ChatMessage.objects.create(message=question, origin='human')
            
            # Process the user's question and get the AI response
            response = rag.answer(question)
            
            response = format_chatbot_output(response)
            # Save the AI response to the chat history
            ChatMessage.objects.create(message=response, origin='AI')
            
            # Redirect to the chatbot page to avoid form resubmission
            return redirect('chatbot')
        # Handle clearing chat history if requested
        if 'clear_history' in request.POST:
            ChatMessage.objects.all().delete()
            return JsonResponse({'success': True})
    # Prepare the context for rendering the template
    context = {'chat_history': chat_history, 'form': form}
    
    # Render the chatbot.html template with the context
    return render(request, 'chatbot.html', context)

def clear_chat_history(request):
    if request.method == 'POST':
        ChatMessage.objects.all().delete()
        return JsonResponse({'success': True})
    else:
        return JsonResponse({'success': False})
