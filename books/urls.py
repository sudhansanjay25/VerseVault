from django.urls import path
from . import views


urlpatterns = [
   
]

urlpatterns = [
    path('login/', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),
    path('admin/dashboard/', views.admin_dashboard, name='admin_dashboard'),
    path('user/dashboard/', views.user_dashboard, name='user_dashboard'),
    path('book/<int:book_id>/review/', views.submit_review, name='submit_review'),
    path('voice-search/', views.voice_search, name='voice_search'),


    path('', views.home, name='home'),
    path("process-visitor-face/", views.process_visitor_face, name="process_visitor_face"),


    # ✅ Admin pages
    path('admin/add-book/', views.add_book_view, name='add_book'),
    path('admin/book-location/', views.assign_book_location, name='assign_book_location'),
    path('admin/visitors/', views.view_visitors, name='view_visitors'),
    path("admin/borrow-history/", views.admin_borrow_history, name="admin_borrow_history"),
    path('admin/reviews/', views.admin_reviews_view, name='admin_reviews'),
    # urls.py
    path("admin/review/delete/<int:review_id>/", views.delete_review, name="delete_review"),
    path("admin/visitors/today/", views.today_visitor_logs, name="today_visitor_logs"),




    # ✅ User features
    path('user/books/', views.view_books, name='view_books'),
    path('user/search/', views.book_search, name='book_search'),
    path('user/borrow-history/', views.user_borrow_history, name='user_borrow_history'),
    path('user/borrow/<int:book_id>/', views.borrow_book, name='borrow_book'),
    path("user/return/<int:record_id>/", views.return_book, name="return_book"),





    #path('story-search/', views.story_search_view, name='story_search'),
    #path('summary-search/', views.summary_search, name='summary_search'),

    #path('borrow/', views.borrow_book, name='borrow_book'),
    #path('return/', views.return_book, name='return_book'),

    #path('recommend/<int:book_id>/', views.book_recommendation, name='book_recommendation'),

    # Future: Sentiment, analytics, fine, happiness
    path('logout/', views.logout_view, name='logout'),
]
