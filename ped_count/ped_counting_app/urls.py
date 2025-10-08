from django.urls import path,include
from .views import add_location,add_pedestrianCount

urlpatterns = [
    path('location/', add_location, name='index'),
    path('pedestrianCount/', add_pedestrianCount, name='add_pedestrianCount'),

]
