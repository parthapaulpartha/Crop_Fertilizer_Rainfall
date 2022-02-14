from django import forms
from django.forms import TextInput, Textarea
from .models import contact

class ContactForm(forms.ModelForm):
    class Meta:
        model = contact
        fields = [
            'name',
            'email',
            'message'
        ]
        widgets = {'name': TextInput(attrs={'placeholder': 'Enter your name'}),
                   'email': TextInput(attrs={'placeholder': 'Enter your email'}),
                   'message': Textarea(attrs={'placeholder': 'Write here your message!'})
                   }
        # widgets = {'email': TextInput(attrs={'placeholder': 'Enter your email'})}