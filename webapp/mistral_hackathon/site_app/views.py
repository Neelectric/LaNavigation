from django.shortcuts import render
from django.template import loader
from django.http import HttpResponse
from main import explain
#form .models import Input, Output

def home(request):
    print("Home page")
    template = loader.get_template('home.html')
    return HttpResponse(template.render({}, request))

def search(request):
    print("Search page")
    context = {}
    print(explain(request.POST))
    return render(request, 'home.html', context=context)