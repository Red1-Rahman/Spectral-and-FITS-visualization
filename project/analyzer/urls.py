from django.urls import path
from . import views

urlpatterns = [
    path('', views.AnalyzerView.as_view(), name='analyzer'),
]