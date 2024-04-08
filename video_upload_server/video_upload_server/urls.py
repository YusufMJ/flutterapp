from django.urls import path
from video_upload import views

urlpatterns = [
    path('upload/', views.upload_video, name='upload_video'),
    path('csrf_token/', views.get_csrf_token, name='get_csrf_token'),
    path('get_processed_video/<str:video_file_name>/', views.get_processed_video, name='get_processed_video'),
]
