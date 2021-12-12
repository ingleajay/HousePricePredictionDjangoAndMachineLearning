
from django.contrib import admin
from django.urls import path
from ml import views
urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name="index"),
    path('predict/', views.predict, name="predict"),
    path('predict/result/', views.result),
    path('analysis/', views.analysis, name="analysis"),
    path('dataset/', views.dataset, name="dataset"),
    path('chart/', views.chart, name="chart"),
]
