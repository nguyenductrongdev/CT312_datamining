from django.urls import path, include

from . import views

urlpatterns = [
    path('', views.index),
    path('predict', views.predict),
    path('predict_csv', views.predict_csv),
    path('download/<path:path>', views.download),
]
