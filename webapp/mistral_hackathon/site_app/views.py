from django.shortcuts import render
from django.template import loader
from django.http import HttpResponse, JsonResponse
#from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
import sys
import os
from .models import AudioFile

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
import main
#form .models import Input, Output


def home(request):
    if request.method == 'POST':
        print("a")
        audio_file = request.FILES.get('audio')
        if audio_file:
            # Save the audio file to the model
            audio = AudioFile.objects.create(audio=audio_file)
            return JsonResponse({'status': 'success', 'audio_id': audio.id})
        else:
            return JsonResponse({'status': 'error', 'message': 'No audio file provided'}, status=400)
    print("Home page")
    template = loader.get_template('home.html')
    return HttpResponse(template.render({}, request))

def search(request):
    print("Search page")
    counter = 10
    while counter > 0:
        try:
            mistral_response = main.explain(request.POST['q'])
            break
        except:
            print("Timed out, retrying")
            counter -= 1
    context = {"url": request.POST['q'], "response": mistral_response['choices'][0]['message']['content']}
    return render(request, 'search.html', context=context)

#recording audio

# load model and processor
#processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
#model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")
#model.config.forced_decoder_ids = None
