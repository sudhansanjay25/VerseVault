from django import forms
from .models import Book
from .models import BookReview

from .models import Visitor
from django.contrib.auth.hashers import make_password

class BookForm(forms.ModelForm):
    class Meta:
        model = Book
        fields = ['title', 'author', 'publisher', 'publication_year', 'summary', 'available_copies']


class LoginForm(forms.Form):
    mobile = forms.CharField(max_length=15, label="Mobile Number")
    password = forms.CharField(widget=forms.PasswordInput)


class VisitorRegistrationForm(forms.ModelForm):
    password = forms.CharField(widget=forms.PasswordInput)

    class Meta:
        model = Visitor
        fields = ['name', 'age', 'mobile', 'email', 'address', 'password', 'role']  # adjust fields as per your model


    def save(self, commit=True):
        visitor = super().save(commit=False)
        visitor.password = make_password(self.cleaned_data['password'])
        if commit:
            visitor.save()
        return visitor


    def save(self, commit=True):
        visitor = super().save(commit=False)
        visitor.password = make_password(self.cleaned_data['password'])  # hash the password
        if commit:
            visitor.save()
        return visitor


class BookReviewForm(forms.ModelForm):
    class Meta:
        model = BookReview
        fields = ['review']
        widgets = {
            'review': forms.Textarea(attrs={
                'placeholder': 'Share your thoughts here... 💬',
                'rows': 5
            }),
        }