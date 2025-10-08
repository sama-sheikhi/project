from rest_framework import serializers
from .models import Location,PedestrianCount,Prediction

class LocationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Location
        fields = '__all__'

class PedestrianCountSerializer(serializers.ModelSerializer):
    class Meta:
        model = PedestrianCount
        fields = '__all__'

class PredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Prediction
        fields = '__all__'

