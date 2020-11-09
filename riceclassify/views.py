from django.shortcuts import render
from django.http import HttpResponse
import pickle

# Create your views here.


def index(request):
    if request.method == "GET":
        return render(request, 'index.html')
    if request.method == "POST":
        a = request.body
        return render(request, a)


def predict(request):
    if request.method == "POST":
        req = list(request.POST.values())[1:]
        req = [float(val) for val in req]
        print(req[0:-1])
        if req[-1] == 1:
            with open('/home/trongnguyen/Dev/DataMining/data_mining_project/CT312_datamining/riceclassify/static/decision_tree_model.sav', 'rb') as f:
                model = pickle.load(f)
            result = model.predict([req[0:-1]])
        elif req[-1] == 2:
            with open('/home/trongnguyen/Dev/DataMining/data_mining_project/CT312_datamining/riceclassify/static/naive_bayes_model.sav', 'rb') as f:
                model = pickle.load(f)
            result = model.predict([req[0:-1]])
        elif req[-1] == 3:
            with open('/home/trongnguyen/Dev/DataMining/data_mining_project/CT312_datamining/riceclassify/static/svm_model.sav', 'rb') as f:
                model = pickle.load(f)
            result = model.predict([req[0:-1]])

        return render(request, "predict.html", {"result": result[0]})
