from django.shortcuts import render, redirect
from django.http import HttpResponse, Http404, JsonResponse, HttpResponseServerError
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
from rest_framework.decorators import api_view
from rest_framework.response import Response

# Create your views here.


@api_view(['GET'])
def predict_rice(request):
    print(request.GET)
    try:
        area = request.GET.get('area').split(',')
        perimeter = request.GET.get('perimeter').split(',')
        majoraxis = request.GET.get('majoraxis').split(',')
        minoraxis = request.GET.get('minoraxis').split(',')
        eccentricity = request.GET.get('eccentricity').split(',')
        convexarea = request.GET.get('convexarea').split(',')
        extent = request.GET.get('extent').split(',')
        data = np.array([area, perimeter, majoraxis, minoraxis,
                         eccentricity, convexarea, extent])
        data = data.T
        print(data)
        columns = ['area', 'perimeter', 'majoraxis', 'minoraxis',
                   'eccentricity', 'convexarea', 'extent']
        data = pd.DataFrame(data=data, columns=columns)
        data = data.astype('float')
        if not None in data:
            # model_path = 'riceclassify/static/model.sav'
            # classifier = pickle.load(open(model_path, 'rb'))
            classifier = getattr(settings, 'MODEL')
            label = classifier.predict(classifier.scaler.transform(data))
            return_data = {
                'error': '0',
                'message': 'Successful',
                'label': label
            }
        else:
            return_data = {
                'error': '1',
                'message': 'Invalid Parameters'
            }
    except Exception as error:
        return_data = {
            'error': '2',
            'message': str(error)
        }
    return Response(return_data)


def predict(request):
    if request.method == "GET":
        req = list(request.GET.values())
        req = [float(val) for val in req]
        print(req)
        filename = os.path.join(os.path.dirname(
            __file__), 'static/model.sav')
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        result = model.predict(model.scaler.transform([req])).tolist()
        return JsonResponse(result, safe=False)


def index(request):
    if request.method == "GET":
        return render(request, 'index.html')
    if request.method == "POST":
        a = request.body
        return render(request, a)


# def predict(request):
#     if request.method == "POST":
#         req = list(request.POST.values())[1:]
#         req = [float(val) for val in req]
#         print(req)
#         filename = os.path.join(os.path.dirname(
#             __file__), 'static/model.sav')
#         with open(filename, 'rb') as f:
#             model = pickle.load(f)
#         result = model.predict(model.scaler.transform([req]))
#         return render(request, "predict.html", {"result": result[0]})


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
            __file__), 'static/model.sav')
        # with open(filename, 'rb') as f:
        #     model = pickle.load(f)
        classifier = getattr(settings, 'MODEL')
        result = dataset.assign(CLASS=classifier.predict(
            classifier.scaler.transform(dataset)))

        path = os.path.join(os.path.dirname(
            __file__), '../media/result_' + uploaded_file_url.split('/')[-1])
        # file=open(djangoSettings.STATIC_ROOT+'/game'+name+'.json','w')
        result.to_csv(path)
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
        res = {'rows': rows, "path": path.split('/')[-1]}

        return JsonResponse(res, safe=False)
        # return render(request, 'predict-csv.html', {'rows': rows,'path': path.split('/')[-1]})


def download(request, path):
    file_path = os.path.join(settings.MEDIA_ROOT, path)
    if os.path.exists(file_path):
        with open(file_path, 'rb') as fh:
            response = HttpResponse(
                fh.read(), content_type="application/vnd.ms-excel")
            response['Content-Disposition'] = 'inline; filename=' + \
                os.path.basename(file_path)
            return response
    raise Http404

# @api_view(['GET'])
# def api(request):
#     if request.method == 'GET':
