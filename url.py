from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('camera/', views.start_camera, name='camera'),
]







detector ----urls.py
