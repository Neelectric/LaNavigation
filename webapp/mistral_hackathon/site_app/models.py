from django.db import models
from datetime import datetime
import os

class Input(models.Model):
    value = models.CharField(max_length=2000)

    def __str__(self):
        return self.name
    
class Output(models.Model):
    input = models.ForeignKey(Input, on_delete=models.CASCADE)
    value = models.CharField(max_length=2000)

    def __str__(self):
        return self.name


def audio_file_name(instance, filename):
    # Get the current timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Extract the file extension
    extension = filename.split('.')[-1]
    
    # Construct the new filename
    new_filename = f"{timestamp}.{extension}"

    # Save the file in the 'audio_files/' directory
    return os.path.join('audio_files', new_filename)

class AudioFile(models.Model):
    audio = models.FileField(upload_to=audio_file_name)  # Use the callable function for file naming
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.audio.name
    
class Answer(models.Model):
    audio = models.ForeignKey(AudioFile, on_delete=models.CASCADE, related_name='answers')
    response_text = models.TextField()  # The answer text
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Answer for AudioFile {self.audio.id} created at {self.created_at}"