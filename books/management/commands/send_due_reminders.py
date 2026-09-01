from django.core.management.base import BaseCommand
from books.models import BorrowRecord
from django.utils import timezone
from django.core.mail import EmailMultiAlternatives
from datetime import timedelta, date

class Command(BaseCommand):
    help = "Send due date reminders for borrowed books"

    def handle(self, *args, **options):
        today = date.today()
        upcoming_due_date = today + timedelta(days=1)

        due_soon = BorrowRecord.objects.filter(
            return_date__isnull=True,
            due_date__lte=upcoming_due_date
        ).select_related("visitor", "book")

        reminders_sent = 0

        for record in due_soon:
            overdue_days = 0
            fine = 0

            text_body = (
                f"Hi {record.visitor.name},\n\n"
                f"Reminder: The book '{record.book.title}' you borrowed is due on {record.due_date}.\n"
                f"Please return it on time to avoid fines."
            )

            html_body = f"""
                <html>
                <body>
                    <p>Hi <strong>{record.visitor.name}</strong>,</p>
                    <p>This is a friendly reminder that your borrowed book <strong>'{record.book.title}'</strong> is due on <strong>{record.due_date}</strong>.</p>
            """

            if today > record.due_date:
                overdue_days = (today - record.due_date).days
                fine = overdue_days * 5
                text_body += f"\n\n⚠️ You're already {overdue_days} day(s) late. A fine of ₹{fine} will apply!"
                html_body += f"""
                    <p style="color:red;"><strong>⚠️ Overdue by {overdue_days} day(s).</strong><br>
                    A fine of ₹{fine} will apply.</p>
                """

            text_body += "\n\nThank you,\nSmart Library"
            html_body += "<br><p>Thank you,<br>Smart Library 📚</p></body></html>"

            # Send email using EmailMultiAlternatives
            subject = "📚 Book Due Reminder"
            from_email = "yourlibraryemail@gmail.com"  # Update this based on EMAIL_HOST_USER
            to_email = record.visitor.email

            email = EmailMultiAlternatives(subject, text_body, from_email, [to_email])
            email.attach_alternative(html_body, "text/html")
            email.send()

            reminders_sent += 1

        self.stdout.write(self.style.SUCCESS(f"{reminders_sent} reminder(s) sent."))
