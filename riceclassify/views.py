from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.


def index(request):
    if request.method == "GET":
        return render(request, 'index.html')
    if request.method == "POST":
        a = request.body
        return render(request, a)


def predict(request):
    if request.method == "POST":
        attr = dict(request.POST.items())
        del attr['csrfmiddlewaretoken']
        print(type(attr))
        # for x, y in attr.items():
        #     print(x, y)
        return HttpResponse(attr)
