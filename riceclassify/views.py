from django.shortcuts import render
from django.http import HttpResponse, Http404
import pickle
import pandas as pd
import io
import csv
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import os
import numpy as np
from io import StringIO
import json
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
        print(req)
        filename = os.path.join(os.path.dirname(
            __file__), 'static/svm_model.sav')
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        result = model.predict([req])
        return render(request, "predict.html", {"result": result[0]})


def predict_csv(request):
    if request.method == 'POST':
        # file = request.FILES['fXTest'].read()
        myfile = request.FILES['fXTest']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        print(myfile.name, uploaded_file_url)
        dataset = open(os.path.join(
            settings.MEDIA_ROOT, myfile.name), 'r').read()
        # dataset = [dataset.split()]
        # for data in dataset:
        #     dta = data.split(',')
        #     print(dta)
        dataset = StringIO(dataset)
        dataset = pd.read_csv(dataset, sep=',')
        # print(dataset)
        filename = os.path.join(os.path.dirname(
            __file__), 'static/svm_model.sav')
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        result = dataset.assign(CLASS=model.predict(dataset))
        filename = os.path.join(os.path.dirname(
            __file__), '../media/result.csv')
        # file=open(djangoSettings.STATIC_ROOT+'/game'+name+'.json','w')
        result.to_csv(filename)
        print(result)
        rows = []
        for i, row in result.iterrows():
            rows.append({
                'AREA': row.AREA,
                'PERIMETER': row.PERIMETER,
                'MAJORAXIS': row.MAJORAXIS,
                'MINORAXIS': row.MINORAXIS,
                'ECCENTRICITY': row.ECCENTRICITY,
                'CONVEX_AREA': row.CONVEX_AREA,
                'EXTENT': row.EXTENT,
                'CLASS': row.CLASS,
            })
        # return HttpResponse(result)
        # json_records = result.reset_index().to_json(orient='records')
        # result = [json.loads(json_records)]
        # context = {'d': result}
        # print(result)
        return render(request, 'predict-csv.html', {'rows': rows})


def download(request, filename):
    file_path = os.path.join(settings.MEDIA_ROOT, filename)
    if os.path.exists(file_path):
        with open(file_path, 'rb') as fh:
            response = HttpResponse(
                fh.read(), content_type="application/vnd.ms-excel")
            response['Content-Disposition'] = 'inline; filename=' + \
                os.path.basename(file_path)
            return response
    raise Http404
