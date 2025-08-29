from django.shortcuts import render, redirect
from django.contrib import messages
from .models import Book
from .forms import BookForm  # we'll create this next
from django.http import HttpResponse
from django.contrib.auth.hashers import check_password
from .forms import LoginForm
from .models import Visitor

from .models import BorrowRecord
from .models import Book, BookLocation

from .forms import VisitorRegistrationForm
from django.db.models import Q
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from django.utils.dateformat import DateFormat
from django.shortcuts import get_object_or_404
from datetime import date, timedelta
from django.utils import timezone
from django.core.mail import send_mail
from django.db.models import F  # 🔁 Add this!


from collections import Counter
from django.db.models.functions import TruncMonth
from datetime import datetime
import calendar
from collections import defaultdict
from django.db.models import Avg

from django.utils.timezone import make_aware
from .forms import VisitorRegistrationForm

from django.contrib.auth.hashers import make_password


from books.models import BorrowRecord
from django.db.models import F
from datetime import date

from books.models import Book
from django.utils.timezone import now



from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import openai, json
from django.conf import settings

from django.db.models import Count, Avg
from .utils import check_if_fake_user  # 🧠 Load your model-based checker

from .utils import is_spam_review
from .forms import BookReviewForm


# books/views.py
import joblib
import os

from .models import BookReview
 # assume you've made a form for BookReview


from django.shortcuts import get_object_or_404
from django.contrib.auth.decorators import login_required  # optional
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import json

# decorators.py
from functools import wraps


from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from .face_recognition import load_known_faces, match_face
from .models import Visitor
import face_recognition
import numpy as np
import base64
import cv2


# books/views.py
from .models import VisitorLog
from django.utils.timezone import now


from django.core.mail import send_mail
from django.utils import timezone
from .models import Visitor, VisitorLog
from .face_recognition import load_known_faces, match_face
import base64, cv2, numpy as np, face_recognition
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse

@csrf_exempt
def process_visitor_face(request):
    if request.method == "POST":
        data = request.POST.get("image")
        if not data:
            return JsonResponse({"error": "No image received"}, status=400)

        try:
            # Decode base64 image
            img_data = base64.b64decode(data.split(',')[1])
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Get face encoding
            face_encodings = face_recognition.face_encodings(rgb_img)
            if not face_encodings:
                return JsonResponse({"error": "No face detected"}, status=400)

            new_encoding = face_encodings[0]
            known_encodings, known_names = load_known_faces()
            matched_name = match_face(new_encoding, known_encodings, known_names)

            # ✅ Match found - Authorized
            if matched_name:
                visitor = Visitor.objects.filter(name=matched_name).first()
                if visitor:
                    VisitorLog.objects.create(
                        name=matched_name,
                        face_encoding=list(new_encoding),
                        status='Authorized'
                    )
                    return JsonResponse({"status": "authorized", "name": matched_name})

            # ❌ Match not found - Unauthorized
            VisitorLog.objects.create(
                name="Unknown",
                face_encoding=list(new_encoding),
                status='Unauthorized'
            )

            # ✅ Send email alert to admin
            send_mail(
                subject="🚨 Unauthorized Visitor Detected",
                message=f"""
An unknown person was detected at the library entrance.

🕒 Time: {timezone.now().strftime('%Y-%m-%d %H:%M:%S')}
📍 Status: Unauthorized
🧾 Action Required: Please check the admin panel > Today's Visitors.

Smart Library AI
""",
                from_email=None,  # Uses DEFAULT_FROM_EMAIL from settings.py
                recipient_list=["sajithjaganathan7@gmail.com"],  # 🔁 Replace with your admin email
                fail_silently=False,
            )

            return JsonResponse({"status": "unauthorized", "name": "Unknown"})

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid method"}, status=405)



def today_visitor_logs(request):
    if request.session.get('role') != 'admin':
        return redirect('login')

    today = now().date()
    logs = VisitorLog.objects.filter(timestamp__date=today).order_by('-timestamp')

    authorized = logs.filter(status='Authorized')
    unauthorized = logs.filter(status='Unauthorized')

    return render(request, "admin/today_visitors.html", {
        "authorized": authorized,
        "unauthorized": unauthorized,
        "date": today
    })





def visitor_required(view_func):
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        visitor_id = request.session.get('visitor_id')
        if not visitor_id:
            return redirect('login')
        request.visitor = Visitor.objects.get(id=visitor_id)
        return view_func(request, *args, **kwargs)
    return wrapper


def login_view(request):
    error = None
    if request.method == "POST":
        form = LoginForm(request.POST)
        if form.is_valid():
            mobile = form.cleaned_data['mobile']
            password = form.cleaned_data['password']
            try:
                visitor = Visitor.objects.get(mobile=mobile)
                if check_password(password, visitor.password):
                    request.session['visitor_id'] = visitor.id
                    request.session['visitor_name'] = visitor.name
                    request.session['role'] = visitor.role
                    return redirect('admin_dashboard' if visitor.role == 'admin' else 'user_dashboard')
                else:
                    error = "Invalid password."
            except Visitor.DoesNotExist:
                error = "Visitor not found."
    else:
        form = LoginForm()
    return render(request, 'login.html', {'form': form, 'error': error})



def register_view(request):
    if request.method == 'POST':
        form = VisitorRegistrationForm(request.POST)
        if form.is_valid():
            visitor = form.save(commit=False)
            visitor.password = make_password(form.cleaned_data['password'])
            visitor.save()
            messages.success(request, 'Registration successful! You can now login.')
            return redirect('login')
    else:
        form = VisitorRegistrationForm()
    return render(request, 'register.html', {'form': form})



def admin_dashboard(request):
    total_books = Book.objects.count()
    active_visitors = Visitor.objects.filter(role='user').count()
    books_borrowed = BorrowRecord.objects.count()
    books_pending_return = BorrowRecord.objects.filter(return_date__isnull=True).count()
    recent_books = Book.objects.order_by('-id')[:1]
    recent_visitors = Visitor.objects.order_by('-id')[:1]
    recent_borrows = BorrowRecord.objects.select_related('book').order_by('-borrow_date')[:1]

    # 👇 Paste this block inside the function
    today = timezone.now().date()
    todays_logs = VisitorLog.objects.filter(timestamp__date=today).order_by('-timestamp')
    authorized = todays_logs.filter(status='Authorized')
    unauthorized = todays_logs.filter(status='Unauthorized')

    context = {
        'total_books': total_books,
        'active_visitors': active_visitors,
        'books_borrowed': books_borrowed,
        'books_pending_return': books_pending_return,
        'recent_books': recent_books,
        'recent_visitors': recent_visitors,
        'recent_borrows': recent_borrows,
        'authorized_visitors': authorized,
        'unauthorized_visitors': unauthorized,
    }
    return render(request, 'admin/a-dashboard.html', context)





def user_dashboard(request):
    if request.session.get('role') != 'user':
        return redirect('login')

    visitor_id = request.session.get('visitor_id')
    visitor = Visitor.objects.get(id=visitor_id)

    # Book stats
    today = date.today()
    on_time = BorrowRecord.objects.filter(visitor=visitor, return_date__isnull=False, return_date__lte=F('due_date')).count()
    late = BorrowRecord.objects.filter(visitor=visitor, return_date__isnull=False, return_date__gt=F('due_date')).count()

    # --- Personal Recommendations ---
    recommendations = []
    borrowed_books = Book.objects.filter(borrowrecord__visitor=visitor).distinct()
    borrowed_embeddings = [b.summary_embedding for b in borrowed_books if b.summary_embedding]
    if borrowed_embeddings:
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        avg_emb = np.mean(np.array(borrowed_embeddings), axis=0)
        candidate_books = Book.objects.exclude(borrowrecord__visitor=visitor).exclude(summary_embedding=None)
        rec_scores = []
        for book in candidate_books:
            book_emb = np.array(book.summary_embedding)
            similarity = cosine_similarity([avg_emb], [book_emb])[0][0]
            rec_scores.append((book, similarity))
        rec_scores.sort(key=lambda x: x[1], reverse=True)
        top_recs = [b for b, s in rec_scores if s > 0.4][:3]
        for book in top_recs:
            try:
                book.location = BookLocation.objects.get(book=book)
            except BookLocation.DoesNotExist:
                book.location = None
        recommendations = top_recs
    else:
        borrowed_authors = borrowed_books.values_list('author', flat=True)
        rec_books = Book.objects.filter(author__in=borrowed_authors).exclude(borrowrecord__visitor=visitor).distinct()[:3]
        for book in rec_books:
            try:
                book.location = BookLocation.objects.get(book=book)
            except BookLocation.DoesNotExist:
                book.location = None
        recommendations = list(rec_books)

    context = {
        "visitor": visitor,
        "books": Book.objects.all(),
        "on_time": on_time,
        "late": late,
        "recommendations": recommendations
    }
    return render(request, "user/u-dashboard.html", context)






# Load the trained spam model
model_path = os.path.join(os.path.dirname(__file__), 'spam_model.pkl')
spam_model = joblib.load(model_path)

def is_spam_review(text):
    return spam_model.predict([text])[0] == 1



def home(request):
    reviews = BookReview.objects.filter(is_spam=False)
    return render(request, 'home.html', {'reviews': reviews})



def add_book_view(request):
    if request.method == 'POST':
        form = BookForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            messages.success(request, "Book added successfully!")
            return redirect('add_book')
    else:
        form = BookForm()

    return render(request, 'admin/add_book.html', {'form': form})


def assign_book_location(request):
    if request.session.get('role') != 'admin':
        return redirect('login')
    books = Book.objects.all()
    if request.method == "POST":
        book_id = request.POST['book']
        block = request.POST['block']
        row = request.POST['row']
        column = request.POST['column']
        position = request.POST['position']
        book = Book.objects.get(id=book_id)
        BookLocation.objects.update_or_create(
            book=book,
            defaults={'block': block, 'row': row, 'column': column, 'position': position}
        )
        return redirect('assign_book_location')
    return render(request, 'admin/book_location.html', {'books': books})




def view_visitors(request):
    visitors = Visitor.objects.all()
    user_count = visitors.filter(role='user').count()
    admin_count = visitors.filter(role='admin').count()
    month_data = visitors.annotate(month=TruncMonth("created_at")).values("month").annotate(count=Count("id")).order_by("month")
    months = [entry["month"].strftime("%b %Y") for entry in month_data if entry["month"]]
    monthly_counts = [entry["count"] for entry in month_data if entry["month"]]
    average_score = round(visitors.aggregate(avg=Avg("score"))["avg"] or 0)
    top_readers = visitors.filter(role='user').order_by("-score")[:5]
    fake_users = [v for v in visitors.filter(role='user') if check_if_fake_user(v.name, v.email, v.address, v.age, v.score, v.visit_count)]
    spam_visitors = Visitor.objects.filter(name="abcde")
    return render(request, "admin/visitors.html", {
        "visitors": visitors,
        "user_count": user_count,
        "admin_count": admin_count,
        "months": months,
        "monthly_counts": monthly_counts,
        "average_score": average_score,
        "top_readers": top_readers,
        "fake_users": fake_users,
        "spam_visitors": spam_visitors,
    })


def submit_review(request, book_id):
    book = get_object_or_404(Book, id=book_id)
    visitor_id = request.session.get('visitor_id')
    if not visitor_id:
        return redirect('login')
    visitor = get_object_or_404(Visitor, id=visitor_id)
    if request.method == "POST":
        form = BookReviewForm(request.POST)
        if form.is_valid():
            review = form.save(commit=False)
            review.book = book
            review.visitor = visitor
            review.sentiment = "Spam" if is_spam_review(review.review) else "Genuine"
            review.save()
            return redirect("user_dashboard")
    else:
        form = BookReviewForm()
    return render(request, "user/submit_review.html", {"form": form, "book": book})



def admin_reviews_view(request):
    if request.session.get('role') != 'admin':
        return redirect('login')
    
    reviews = BookReview.objects.select_related('book', 'visitor').order_by('-created_at')
    
    return render(request, 'admin/reviews.html', {
        'reviews': reviews
    })



def view_books(request):
    if request.session.get('role') != 'user':
        return redirect('login')

    books = Book.objects.all().select_related()
    book_locations = {loc.book_id: loc for loc in BookLocation.objects.all()}
    
    return render(request, 'user/view_books.html', {
        'books': books,
        'locations': book_locations
    })

# views.py
from django.core.paginator import Paginator

def all_reviews(request):
    reviews = BookReview.objects.select_related('book', 'visitor').order_by('-created_at')

    # Filters
    filter_type = request.GET.get("filter")
    if filter_type == "spam":
        reviews = reviews.filter(is_spam=True)
    elif filter_type == "negative":
        reviews = reviews.filter(sentiment="Negative")

    paginator = Paginator(reviews, 10)  # 10 reviews per page
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    return render(request, "admin/review_moderation.html", {"page_obj": page_obj})



# views.py
from django.contrib.admin.views.decorators import staff_member_required

@staff_member_required
def delete_review(request, review_id):
    review = get_object_or_404(BookReview, id=review_id)
    review.delete()
    messages.success(request, "Review deleted successfully!")
    return redirect("all_reviews")


model = SentenceTransformer('all-MiniLM-L6-v2')

from .models import Book, BookLocation

def book_search(request):
    if request.session.get('role') != 'user':
        return redirect('login')

    query = request.GET.get('q')
    results = []
    recommendations = []

    visitor_id = request.session.get('visitor_id')
    visitor = None
    if visitor_id:
        try:
            visitor = Visitor.objects.get(id=visitor_id)
        except Visitor.DoesNotExist:
            visitor = None

    if query:
        # Try embedding-based search first
        try:
            query_emb = model.encode(query)
            all_books = Book.objects.exclude(summary_embedding=None)
            scores = []
            for book in all_books:
                book_emb = np.array(book.summary_embedding)
                similarity = cosine_similarity([query_emb], [book_emb])[0][0]
                scores.append((book, similarity))
            scores.sort(key=lambda x: x[1], reverse=True)
            filtered_books = [b for b, s in scores if s > 0.4][:5]
            # Attach location to each book
            for book in filtered_books:
                try:
                    book.location = BookLocation.objects.get(book=book)
                except BookLocation.DoesNotExist:
                    book.location = None
            results = filtered_books
        except Exception:
            results = []

        # Fallback: If no embedding results, do a simple filter on all relevant fields
        if not results:
            fallback_books = Book.objects.filter(
                Q(title__icontains=query) |
                Q(author__icontains=query) |
                Q(summary__icontains=query) |
                Q(publisher__icontains=query) |
                Q(publication_year__icontains=query)
            )
            # Attach location to each book
            for book in fallback_books:
                try:
                    book.location = BookLocation.objects.get(book=book)
                except BookLocation.DoesNotExist:
                    book.location = None
            results = list(fallback_books)

    # --- Personal Recommendations ---
    if visitor:
        # Get books the user has borrowed
        borrowed_books = Book.objects.filter(borrowrecord__visitor=visitor).distinct()
        borrowed_embeddings = [b.summary_embedding for b in borrowed_books if b.summary_embedding]
        if borrowed_embeddings:
            # Average embedding of borrowed books
            avg_emb = np.mean(np.array(borrowed_embeddings), axis=0)
            # Recommend books not yet borrowed, most similar to user's history
            candidate_books = Book.objects.exclude(borrowrecord__visitor=visitor).exclude(summary_embedding=None)
            rec_scores = []
            for book in candidate_books:
                book_emb = np.array(book.summary_embedding)
                similarity = cosine_similarity([avg_emb], [book_emb])[0][0]
                rec_scores.append((book, similarity))
            rec_scores.sort(key=lambda x: x[1], reverse=True)
            top_recs = [b for b, s in rec_scores if s > 0.4][:3]
            for book in top_recs:
                try:
                    book.location = BookLocation.objects.get(book=book)
                except BookLocation.DoesNotExist:
                    book.location = None
            recommendations = top_recs
        else:
            # Fallback: recommend books by same author as previously borrowed
            borrowed_authors = borrowed_books.values_list('author', flat=True)
            rec_books = Book.objects.filter(author__in=borrowed_authors).exclude(borrowrecord__visitor=visitor).distinct()[:3]
            for book in rec_books:
                try:
                    book.location = BookLocation.objects.get(book=book)
                except BookLocation.DoesNotExist:
                    book.location = None
            recommendations = list(rec_books)

    return render(request, 'user/book_search.html', {
        'query': query,
        'results': results,
        'recommendations': recommendations
    })

# views.py
from django.http import JsonResponse
from vosk import Model, KaldiRecognizer
import sounddevice as sd
import queue
import json

q = queue.Queue()

def voice_search(request):
    model = Model("vosk-model-small-en-us-0.15")  # your model path
    rec = KaldiRecognizer(model, 16000)

    def callback(indata, frames, time, status):
        if status:
            print("🔴 Status:", status)
        q.put(bytes(indata))

    try:
        with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                            channels=1, callback=callback):
            print("🎤 Listening for 5 seconds...")
            collected = b''
            while len(collected) < 5 * 16000 * 2:  # collect 5 sec
                data = q.get()
                collected += data
                if rec.AcceptWaveform(data):
                    break
            result = json.loads(rec.FinalResult())
            print("✅ Final:", result)
            return JsonResponse({"query": result.get("text", "")})
    except Exception as e:
        print("❌ Error:", str(e))
        return JsonResponse({"error": str(e)}, status=500)# views.py






def borrow_book(request, book_id):
    if request.session.get('role') != 'user':
        return redirect('login')

    visitor_id = request.session.get('visitor_id')
    visitor = get_object_or_404(Visitor, id=visitor_id)
    book = get_object_or_404(Book, id=book_id)

    already_borrowed = BorrowRecord.objects.filter(visitor=visitor, book=book, return_date__isnull=True).exists()
    
    if already_borrowed:
        message = "You already borrowed this book. Return it before borrowing again."
    elif book.available_copies <= 0:
        message = "Sorry, this book is not available right now."
    else:
        # ✅ Create borrow record
        BorrowRecord.objects.create(
            visitor=visitor,
            book=book,
            due_date=date.today() + timedelta(days=7)
        )


        send_mail(
            subject="📚 Book Borrowed Successfully",
            message=f"""
        Hello {visitor.name},

        You have successfully borrowed: "{book.title}" by {book.author}.
        🗓️ Borrowed On: {date.today()}
        📅 Due Date: {date.today() + timedelta(days=7)}

        Please make sure to return the book by the due date to avoid fines.

        Thank you,
        Smart Library 📖
        """,
            from_email=None,  # Uses DEFAULT_FROM_EMAIL from settings.py
            recipient_list=[visitor.email],
            fail_silently=False
        )


        # ✅ Update book stats
        book.available_copies -= 1
        book.borrow_count += 1
        book.save()

        # ✅ Update visitor stats
        visitor.visit_count += 1
        visitor.score += 2   # 🎯 Add points for borrowing
        visitor.save()

        message = "Book borrowed successfully."

    return render(request, "user/u-dashboard.html", {
        "message": message,
        "books": Book.objects.all()
    })




def user_borrow_history(request):
    visitor_id = request.session.get('visitor_id')
    if not visitor_id:
        return redirect('login')

    records = BorrowRecord.objects.filter(visitor_id=visitor_id).order_by('-borrow_date')

    today = date.today()
    for r in records:
        r.is_overdue = (r.return_date is None and today > r.due_date)

    return render(request, 'user/borrow_history_chart.html', {
        'records': records,
    })


def return_book(request, record_id):
    if request.session.get("role") != "user":
        return redirect("login")

    visitor_id = request.session.get("visitor_id")
    record = get_object_or_404(BorrowRecord, id=record_id, visitor_id=visitor_id)
    visitor = record.visitor  # Needed for scoring

    if record.return_date is None:
        today = date.today()
        record.return_date = today
        record.save()

        # ✅ Update book copies
        record.book.available_copies += 1
        record.book.save()

        # ✅ Scoring logic
        if today > record.due_date:
            days_late = (today - record.due_date).days
            fine = days_late * 5
            visitor.score -= 5  # Penalty for late return
            messages.warning(request, f"You returned late by {days_late} day(s). Fine: ₹{fine}")
        else:
            visitor.score += 2  # Bonus for on-time return
            messages.success(request, f"You returned '{record.book.title}' successfully.")

        visitor.save()

        # ✅ Return Confirmation Email
        return_message = f"""
Hi {visitor.name},

✅ You have successfully returned the book "{record.book.title}".

🧾 Borrowed On: {record.borrow_date}
📅 Returned On: {record.return_date}
📖 Book: {record.book.title}

{f"⚠️ Returned {days_late} day(s) late. Fine applied: ₹{fine}" if today > record.due_date else "🎉 Thank you for returning the book on time!"}

Keep reading and earn more points!

📊 Your Current Score: {visitor.score}

Regards,  
Smart Library 📘
"""
        send_mail(
            subject="✅ Book Returned Successfully",
            message=return_message,
            from_email=None,
            recipient_list=[visitor.email],
            fail_silently=False,
        )

        # ✅ Optional: Also send Due Reminder format (if late)
        if today > record.due_date:
            due_message = f"""
Hi {visitor.name},

📌 This is a reminder that the book "{record.book.title}" you borrowed was due on {record.due_date}.

🧾 Borrowed On: {record.borrow_date}
📅 Due Date: {record.due_date}
⚠️ You returned it {days_late} day(s) late. A fine of ₹{fine} has been applied.

Please make sure to return books on time in the future to maintain a good score.

Thank you,  
Smart Library 📚
"""
            send_mail(
                subject="📌 Book Due Reminder - Fine Applied",
                message=due_message,
                from_email=None,
                recipient_list=[visitor.email],
                fail_silently=False,
            )

    return redirect("user_dashboard")



def admin_borrow_history(request):
    if request.session.get("role") != "admin":
        return redirect("login")

    # Filters
    selected_month = int(request.GET.get("month", date.today().month))
    selected_year = int(request.GET.get("year", date.today().year))

    # Borrow history
    history = BorrowRecord.objects.select_related("visitor", "book").filter(
        borrow_date__month=selected_month,
        borrow_date__year=selected_year
    ).order_by("-borrow_date")

    # Popular books in selected month
    popular_books = (
        history
        .values("book__title")
        .annotate(count=Count("book"))
        .order_by("-count")[:5]
    )

    # Monthly trend (line chart)
    monthly_trend = (
        BorrowRecord.objects
        .filter(borrow_date__year=selected_year)
        .annotate(month=Count('borrow_date__month'))
        .values("borrow_date__month")
        .annotate(count=Count("id"))
        .order_by("borrow_date__month")
    )

    trend_labels = []
    trend_data = []
    for m in range(1, 13):
        trend_labels.append(calendar.month_abbr[m])
        found = next((item for item in monthly_trend if item["borrow_date__month"] == m), None)
        trend_data.append(found["count"] if found else 0)

    months = [(i, calendar.month_name[i]) for i in range(1, 13)]
    years = [y for y in range(date.today().year - 5, date.today().year + 1)]

    return render(request, "admin/borrow_history.html", {
        "history": history,
        "popular_books": [{"title": b["book__title"], "count": b["count"]} for b in popular_books],
        "months": months,
        "years": years,
        "selected_month": selected_month,
        "selected_year": selected_year,
        "trend_labels": trend_labels,
        "trend_data": trend_data,
    })




@csrf_exempt
def log_face_detection(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        FaceLog.objects.create(
            name=data.get("name", "Unknown"),
            status=data["status"]
        )
        return JsonResponse({"message": "Logged successfully"})




def frequent_questions_view(request):
    # Example frequent questions and answers
    faqs = [
        {"question": "How do I borrow a book?", "answer": "Go to the dashboard or search page, find your book, and click the 'Borrow' button."},
        {"question": "How long can I keep a borrowed book?", "answer": "The standard borrowing period is 7 days. Please return on or before the due date."},
        {"question": "How do I return a book?", "answer": "Go to your borrow history and click 'Return' next to the book you want to return."},
        {"question": "What happens if I return a book late?", "answer": "A fine is applied and your score is reduced. Please return books on time to avoid penalties."},
        {"question": "How are recommendations generated?", "answer": "Recommendations are based on your borrowing history and book similarities."},
    ]
    return render(request, 'user/frequent_questions.html', {'faqs': faqs})


def logout_view(request):
    request.session.flush()
    return redirect('login')
