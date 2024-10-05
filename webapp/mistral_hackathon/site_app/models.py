from django.db import models

class Input(models.Model):
    value = models.CharField(max_length=2000)

    def __str__(self):
        return self.name
    
class Output(models.Model):
    input = models.ForeignKey(Input, on_delete=models.CASCADE)
    value = models.CharField(max_length=2000)

    def __str__(self):
        return self.name
    
class Transcription(models.Model):
    audio_file = models.FileField(upload_to='audio/')
    text = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.text[:50]