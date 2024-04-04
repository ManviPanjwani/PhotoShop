# photoshop_app/urls.py

from django.urls import path

from . import views
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('scan/', views.scan_and_send_email, name='scan_and_send_email'),
    path('register/', views.register_user, name='register'),
    path('registration_success/', views.registration_success, name='registration_success'),
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
