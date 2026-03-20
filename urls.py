from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('start-camera/', views.start_camera, name='start_camera'),
    path('stop-camera/', views.stop_camera, name='stop_camera'),
]
