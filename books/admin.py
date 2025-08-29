from django.contrib import admin
from .models import Visitor, Book, BookLocation, BorrowRecord, BookReview, AdminProfile


class BookLocationInline(admin.StackedInline):
    model = BookLocation
    extra = 0

@admin.register(Book)
class BookAdmin(admin.ModelAdmin):
    list_display = ('title', 'author', 'available_copies', 'borrow_count')
    inlines = [BookLocationInline]


from .models import VisitorLog

@admin.register(VisitorLog)
class VisitorLogAdmin(admin.ModelAdmin):
    list_display = ['name', 'status', 'timestamp']
    list_filter = ['status', 'timestamp']


# Other models
admin.site.register(Visitor)
admin.site.register(BookLocation)
admin.site.register(BorrowRecord)
admin.site.register(BookReview)
admin.site.register(AdminProfile)
