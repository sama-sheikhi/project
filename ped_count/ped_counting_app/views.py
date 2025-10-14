import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"
from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
import numpy as np
from rest_framework.views import APIView
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .models import Location,PedestrianCount,Prediction
from .serializer import LocationSerializer,PedestrianCountSerializer,PredictionSerializer
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from rest_framework.decorators import api_view
from rest_framework.response import Response
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi

# ---------------------------------------------------------------------------------------------------------------------
@swagger_auto_schema(
    method='post',
    operation_summary="Upload CSV data",
    operation_description="Upload a CSV file containing pedestrian count data and save it into the database.",
    manual_parameters=[],
    request_body=openapi.Schema(
        type=openapi.TYPE_OBJECT,
        required=['file'],
        properties={
            'file': openapi.Schema(
                type=openapi.TYPE_STRING,
                format='binary',
                description='CSV file containing pedestrian count data'
            ),
        },
    ),
    responses={
        201: openapi.Response(
            description="Data uploaded successfully",
            examples={
                "application/json": {"message": "100 records processed successfully."}
            }
        ),
        400: openapi.Response(
            description="Invalid file or data format",
            examples={
                "application/json": {"error": "No file uploaded."}
            }
        ),
    }
)

@api_view(['POST'])
def upload_data(request):
    try:
        file = request.FILES['file']
        if not file:
            return Response({'error': 'No file uploaded.'}, status=status.HTTP_400_BAD_REQUEST)
        dt = pd.read_csv(file)

        data = []
        for _, row in dt.iterrows():
            location, _ = Location.objects.get_or_create(
                Sensor_ID=row['Sensor_ID'],
                # Sensor_Name=row['Sensor_Name'],
                defaults={'Sensor_Name': row['Sensor_Name']}

            )

            data.append(PedestrianCount(
                location=location,
                Date_Time=row['Date_Time'],
                Year=row['Year'] if pd.notna(row['Year']) else None,
                Month=row['Month'] if pd.notna(row['Month']) else None,
                Mdate=row['Mdate'] if pd.notna(row['Mdate']) else None,
                Day=row['Day'] if pd.notna(row['Day']) else None,
                Time=row['Time'] if pd.notna(row['Time']) else None,
                Hourly_Counts=row['Hourly_Counts'] if pd.notna(row['Hourly_Counts']) else None,
                total_count=row['total_count'] if pd.notna(row['total_count']) else None
            ))


        PedestrianCount.objects.bulk_create(data, ignore_conflicts=True)

        return Response(
            {"message": f"{len(data)} records processed successfully."},
            status=status.HTTP_201_CREATED
        )
    except Exception as e:
        return Response({"message": str(e)}, status=status.HTTP_400_BAD_REQUEST)

# ---------------------------------------------------------------------------------------------------------------------
# تعداد ساعاتی که برای پیش‌بینی استفاده می‌کنیم
n_steps = 24

@swagger_auto_schema(
    method='post',
    operation_summary="Train LSTM model",
    operation_description="Trains an LSTM model on existing pedestrian count data and saves the model and scaler files.",
    responses={
        200: openapi.Response(
            description="Model trained successfully",
            examples={"application/json": {"message": "Model trained successfully!"}}
        ),
        400: openapi.Response(
            description="No data available for training",
            examples={"application/json": {"message": "No data found for training."}}
        ),
        500: openapi.Response(
            description="Internal server error during model training",
            examples={"application/json": {"error": "Some error message"}}
        ),
    }
)


@api_view(['POST'])
def train_model(request):
    try:
        from tensorflow.keras.models import Sequential, load_model
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
        data = PedestrianCount.objects.all().values()
        dt = pd.DataFrame(data)

        if dt.empty:
            return Response({"message": "No data found for training."}, status=status.HTTP_400_BAD_REQUEST)


        month_map = {'January': 1, 'February': 2, 'March': 3, 'April': 4,
                     'May': 5, 'June': 6, 'July': 7, 'August': 8,
                     'September': 9, 'October': 10, 'November': 11, 'December': 12}

        day_map = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3,
                   'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}

        dt['Month_num'] = dt['Month'].map(month_map)
        dt['Day_num'] = dt['Day'].map(day_map)
        # مرتب‌سازی بر اساس زمان
        dt = dt.sort_values(by="Time")

        # X = dt[['Year', 'Month_num', 'Mdate', 'Day_num', 'Time']]
        # Y = dt['Hourly_Counts']

        features = ['Hourly_Counts', 'Month_num', 'Day_num', 'Time']
        data_values = dt[features].values

        # نرمال‌سازی
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data_values)

        # ساخت sequence
        X, y = [], []
        for i in range(n_steps, len(data_scaled)):
            X.append(data_scaled[i - n_steps:i])
            y.append(data_scaled[i, 0])
        X, y = np.array(X), np.array(y)

        # مدل LSTM
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.callbacks import EarlyStopping

        model = Sequential()
        model.add(LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
        model.add(Dropout(0.2))
        model.add(LSTM(32))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')

        es = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        model.fit(X, y, epochs=30, batch_size=16, callbacks=[es], verbose=1)

        model.save("lstm_model.keras")
        joblib.dump(scaler, "lstm_scaler.pkl")
        # ذخیره مدل
        joblib.dump({'model': model, 'scaler': scaler}, 'model.pkl')

        return Response({'message': 'Model trained successfully!'}, status=status.HTTP_200_OK)

    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# ---------------------------------------------------------------------------------------------------------------------
@swagger_auto_schema(
    method='get',
    operation_summary="Predict future pedestrian counts",
    operation_description=(
        "Predicts future pedestrian counts for the specified location using a trained LSTM model. "
        "You can specify how many hours ahead to predict using the `hours` query parameter."
    ),
    manual_parameters=[
        openapi.Parameter(
            'hours', openapi.IN_QUERY,
            description="Number of hours to predict ahead (default: 24)",
            type=openapi.TYPE_INTEGER,
            required=False
        ),
        openapi.Parameter(
            'location', openapi.IN_QUERY,
            description="Sensor or location name to predict for",
            type=openapi.TYPE_STRING,
            required=True
        ),
    ],
    responses={
        200: openapi.Response(
            description="Prediction completed successfully",
            examples={
                "application/json": {
                    "message": "Prediction for the next 24 hours completed successfully.",
                    "predictions": [105.2, 110.4, 120.1],
                    "r2_score": 0.87,
                    "mae": 15.4,
                    "rmse": 22.1
                }
            }
        ),
        400: openapi.Response(
            description="Missing or invalid parameters",
            examples={"application/json": {"error": "location parameter required!"}}
        ),
        404: openapi.Response(
            description="Not enough data for prediction",
            examples={"application/json": {"error": "Not enough data for prediction."}}
        ),
        500: openapi.Response(
            description="Internal server error during prediction",
            examples={"application/json": {"error": "Some error message"}}
        ),
    }
)

@api_view(['GET'])
def predict(request):
    try:
        from tensorflow.keras.models import load_model
        hours = int(request.GET.get('hours', 24))
        location_name = request.GET.get('location')
        if not location_name:
            return Response({"error": "location parameter required!"}, status=status.HTTP_400_BAD_REQUEST)

        location = Location.objects.get(Sensor_Name=location_name)

        model = load_model("lstm_model.keras")
        scaler = joblib.load("lstm_scaler.pkl")

        recent_data = PedestrianCount.objects.filter(location=location).order_by('-id')[:hours+24]
        df_recent = pd.DataFrame(list(recent_data.values()))
        if df_recent.empty or len(df_recent) < 24:
            return Response({'error': 'Not enough data for prediction.'}, status=status.HTTP_404_NOT_FOUND)

        month_map = {'January':1,'February':2,'March':3,'April':4,'May':5,'June':6,
                     'July':7,'August':8,'September':9,'October':10,'November':11,'December':12}
        day_map = {'Monday':1,'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5,'Saturday':6,'Sunday':7}

        df_recent['Month_num'] = df_recent['Month'].map(month_map)
        df_recent['Day_num'] = df_recent['Day'].map(day_map)

        features = ['Hourly_Counts', 'Month_num', 'Day_num', 'Time']
        data_values = df_recent[features].values
        data_scaled = scaler.transform(data_values)

        # ساخت sequence برای LSTM
        X_new = []
        y_true_scaled = []
        n_steps = 24
        for i in range(n_steps, len(data_scaled)):
            X_new.append(data_scaled[i-n_steps:i])
            y_true_scaled.append(data_scaled[i,0])
        X_new, y_true_scaled = np.array(X_new), np.array(y_true_scaled)

        y_pred_scaled = model.predict(X_new)

        # inverse scale برای برگردوندن مقادیر
        y_pred = scaler.inverse_transform(np.hstack([y_pred_scaled, np.zeros((len(y_pred_scaled), data_scaled.shape[1]-1))]))[:,0]
        y_true = scaler.inverse_transform(np.hstack([y_true_scaled.reshape(-1,1), np.zeros((len(y_true_scaled), data_scaled.shape[1]-1))]))[:,0]

        # محاسبه متریک‌ها
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred)

        # ذخیره پیش‌بینی‌ها
        for pred_value in y_pred:
            Prediction.objects.create(
                data=location,
                model_name='LSTM',
                predicted_count=float(pred_value)
            )

        return Response({
            "message": f"Prediction for the next {hours} hours completed successfully.",
            "predictions": y_pred.tolist(),
            "r2_score": r2,
            "mae": mae,
            "rmse": rmse
        }, status=status.HTTP_200_OK)

    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)



