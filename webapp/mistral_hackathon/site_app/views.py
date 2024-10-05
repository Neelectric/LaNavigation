from django.shortcuts import render
from django.template import loader
from django.http import HttpResponse, JsonResponse
from django.conf import settings
import datetime
from gtts import gTTS
#from transformers import WhisperProcessor, WhisperForConditionalGeneration
#from datasets import load_dataset
import whisper
import sys
import os
from .models import AudioFile, Answer
import time

from threading import Thread

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
import main
#form .models import Input, Output

#processing audio

model = whisper.load_model("turbo")

def read_audio(audio_file):
    result = model.transcribe(audio_file, language="en", condition_on_previous_text=False, verbose=True)
    return result['segments'][-1]['text']    

def home(request):
    context = {"messages":[{"owner":"otuput", "content":"Hello, how can I help you?"}]}
    if request.method == 'POST':
        audio_file = request.FILES.get('audio')
        if audio_file:
            # Save the audio file to the model (the model will handle renaming it)
            audio = AudioFile.objects.create(audio=audio_file)
            
            # Use the correct file path
            audio_file_path = os.path.join(settings.MEDIA_ROOT, str(audio.audio))

            # If you have a function that processes the audio file
            input_text = read_audio(audio_file_path)
            new_message = {"owner":"input", "content":input_text}
            context["messages"].append(new_message)

            # Start the async task to process the `answer()`
            thread = Thread(target=answer_async, args=(input_text, audio.id))
            thread.start()  # Start async processing in the background

            return JsonResponse({
                'status': 'success', 
                'audio_id': audio.id,
                'new_messages': [new_message]  # Send back the new messages
            })
        return JsonResponse({'status': 'error', 'message': 'No audio file provided'}, status=400)

    print("Home page")
    return render(request, 'index.html', context=context)

def answer(text):
    return "I don't know yet, ask Titas"

def answer_async(text, audio_id):
    """Async function to process the answer in the background."""
    time.sleep(5)  # Simulate delay (replace with actual answer() function call)
    
    # Call the `answer()` function to get the response text
    response_text = answer(text)
    
    # Create a new Answer object and link it to the corresponding AudioFile
    audio = AudioFile.objects.get(id=audio_id)
    Answer.objects.create(audio=audio, response_text=response_text)

def get_answer(request, audio_id):
    """API endpoint for polling the answer."""
    try:
        audio = AudioFile.objects.get(id=audio_id)
        answer = Answer.objects.filter(audio=audio).first()
        
        if answer:
            return JsonResponse({
                'status': 'success',
                'answer_message': {"owner": "output", "content": answer.response_text}
            })
        else:
            return JsonResponse({'status': 'pending'})  # Answer is still processing
    except AudioFile.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': 'Audio file not found'}, status=404)


def search(request):
    print("Search page")
    counter = 10
    while counter > 0:
        try:
            mistral_response = main.explain(request.POST['q'])
            context = {"url": request.POST['q'], "response": mistral_response['choices'][0]['message']['content']}
            break
        except:
            print("Timed out, retrying")
            counter -= 1
    
    return render(request, 'search.html', context=context)

def process_text(request):
    if request.method == 'POST':
        # Get the text input from the form
        text_input = request.POST.get('text_input')

        # Convert the text to speech using gTTS
        tts = gTTS(text_input, lang='en')
        
        # Define the file path to save the audio
        file_path = os.path.join('static', 'audio', 'speech.mp3')
        
        # Save the generated speech audio
        tts.save(file_path)

        # Return the URL of the audio file to be played in the frontend
        audio_url = '/static/audio/speech.mp3'
        return JsonResponse({'audio_url': audio_url})

    return JsonResponse({'error': 'Invalid request'}, status=400)
