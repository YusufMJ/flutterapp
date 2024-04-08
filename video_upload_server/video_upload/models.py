from django.db import models

class Video(models.Model):
    video_file = models.FileField(upload_to='videos/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        app_label = 'video_upload'
