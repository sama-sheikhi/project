from django.urls import path,include
from .views import upload_data, train_model,predict

urlpatterns = [
    path('api/upload-data/', upload_data, name='upload-data'),
    path('api/train-model/', train_model, name='train-model'),
    path('api/predict/', predict, name='predict'),

]

