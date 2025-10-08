from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
from rest_framework.views import APIView
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .models import Location,PedestrianCount,Prediction
from .serializer import LocationSerializer,PedestrianCountSerializer,PredictionSerializer

# @api_view(['GET'])
# def get_location(request):
#     locations = Location.objects.all()
#     serializer = LocationSerializer(locations, many=True)
#     return Response(serializer.data)


@api_view(['POST'])
def add_location(request):
    serializer = LocationSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save()
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)



@api_view(['POST'])
def add_pedestrianCount(request):
    try:
        file = request.FILES['file']
        dt = pd.read_csv(file)

        data = []

        for _, row in dt.iterrows():
            location, _ = Location.objects.get_or_create(
                Sensor_ID=row['Sensor_ID'],
                Sensor_Name=row['Sensor_Name'],
            )

            ped_count, _ = PedestrianCount.objects.get_or_create(
                location=location,
                Date_Time=row['Date_Time'],
                Year=row['Year'],
                Month=row['Month'],
                Mdate=row['Mdate'],
                Day=row['Day'],
                Time=row['Time'],
                Hourly_Counts=row['Hourly_Counts'],
                total_count=int(row['total_count']),
            )

            data.append(ped_count)

        PedestrianCount.objects.bulk_create(data)

        return Response({"message": "Data imported successfully!"}, status=status.HTTP_201_CREATED)

    except Exception as e:
        return Response({"message": str(e)}, status=status.HTTP_400_BAD_REQUEST)
