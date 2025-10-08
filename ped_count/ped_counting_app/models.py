from django.db import models

class Location(models.Model):
    Sensor_ID = models.IntegerField(null=True)
    Sensor_Name = models.CharField(max_length=150, null=True)

    def __str__(self):
        return f"{self.Sensor_Name} - {self.Sensor_ID}  "

class PedestrianCount(models.Model):
    location = models.ForeignKey(Location, on_delete=models.CASCADE, related_name='pedestrian_counts')
    Date_Time = models.CharField(max_length=150)
    Year = models.IntegerField(null=True)
    Month = models.CharField(max_length=150)
    Mdate = models.IntegerField(null=True)
    Day = models.CharField(max_length=150)
    Time = models.IntegerField(null=True)
    Hourly_Counts = models.IntegerField(null=True)
    total_count = models.IntegerField(null=True)

    def __str__(self):
        return f"{self.location.Sensor_Name} - {self.Year} / {self.Month} / {self.Day} - {self.Time}"

class Prediction(models.Model):
    data = models.ForeignKey(Location, on_delete=models.CASCADE, related_name='predictions')
    model_name = models.CharField(max_length=150)
    predicted_count = models.FloatField(null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Prediction for {self.data.Sensor_Name} ({self.model_name})"

