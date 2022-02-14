"""predict_system URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
#from django.contrib import admin
# from django.conf.urls import url
from django.urls import path
from . import views

urlpatterns = [
    path(r'home/', views.home, name='home'),

    # Crop Recommendation Starts from here
    path(r'crop_recommendation/', views.crop_recommendation_predict, name='crop_recommendation'),
    path(r'crop_recommendation/result/', views.crop_recommendation_result, ),
    # Crop Recommendation End here

    # Fertilizer Prediction Starts from here
    path(r'fertilizer_prediction/', views.fertilizer_predict, name='fertilizer_prediction'),
    path(r'fertilizer_prediction/result/', views.fertilizer_predict_result, ),
    # Fertilizer Prediction End here 

    # RainFall Prediction Starts from here
    path(r'rainfall_prediction/', views.rainfall_predict, name='rainfall_prediction'),
    path(r'rainfall_prediction/result/', views.rainfall_result, ),
    # RainFall Prediction End here   

    # About us start
    path(r'about_us/', views.about_us, name='about_us')
]
