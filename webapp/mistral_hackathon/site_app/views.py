from django.shortcuts import render
from django.template import loader
from django.http import HttpResponse, JsonResponse
#from transformers import WhisperProcessor, WhisperForConditionalGeneration
#from datasets import load_dataset
import whisper
import sys
import os
from .models import AudioFile

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
import main
#form .models import Input, Output

#processing audio

# load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")
model.config.forced_decoder_ids = None
model = model.load_model("turbo")

def read_audio(audio_file):
    audio = model.load_audio("audio.mp3")
    audio = model.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = model.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    # decode the audio
    options = model.DecodingOptions()
    result = model.decode(model, mel, options)

    # print the recognized text
    print(result.text)


def home(request):
    if request.method == 'POST':
        audio_file = request.FILES.get('audio')
        if audio_file:
            # Save the audio file to the model
            audio = AudioFile.objects.create(audio=audio_file)
            read_audio(audio_file)

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


