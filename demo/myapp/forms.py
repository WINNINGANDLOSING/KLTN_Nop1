from django import forms
from django.core.validators import EmailValidator, RegexValidator, MinLengthValidator

class QuestionForm(forms.Form):
    question = forms.CharField(
        label='Ask a question',
        max_length=300,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Đặt câu hỏi tại đây',
            'style': (
                'height: 50px; margin-right: 5px; font-size: 16px; border-radius: 25px; align-self: end'
                'border: 2px solid #ced4da; padding: 10px 20px; box-shadow: 0 2px 2px rgba(0, 0, 0, 0.1);'
            )
        }))

class ContactForm(forms.Form):
    name = forms.CharField(max_length=100, validators=[MinLengthValidator(2)])
    email = forms.EmailField(validators=[EmailValidator(message="Enter a valid email address.")])
    phone = forms.CharField(
        max_length=10,
        validators=[
            RegexValidator(
                regex=r'^0\d{9}$',
                message="Phone number must be entered in the format: '0xxxxxxxxx'. Exactly 10 digits starting with 0 are allowed."
            )
        ],
        required=False
    )
    subject = forms.CharField(max_length=100)
    message = forms.CharField(widget=forms.Textarea, validators=[MinLengthValidator(10)])