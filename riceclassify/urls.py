from django.urls import path

from . import views

urlpatterns = [
    path('', views.index),
    path('predict', views.predict),
    path('predict_csv', views.predict_csv),
    path('download/<path:filename>', views.download)
]
