# import joblib
# import numpy as np
# import pandas as pd
# import xgboost as xgb
# # from keras.layers import LSTM
# # -------------------------------------------------load models----------------------------------------------------------
# scaler = joblib.load("models/scaler.pkl")
# linear_model = joblib.load("models/linear_model.pkl")
# rf_model = joblib.load("models/random_forest_model.pkl")
# gb_model = joblib.load("models/gradient_boosting_model.pkl")
# xgb_model = joblib.load("models/xgb_model.pkl")
# lstm_model = joblib.load("ped_count/lstm_model.keras")
#
# month_map = {
#     'January': 1, 'February': 2, 'March': 3, 'April': 4,
#     'May': 5, 'June': 6, 'July': 7, 'August': 8,
#     'September': 9, 'October': 10, 'November': 11, 'December': 12
# }
#
# day_map = {
#     'Monday': 1, 'Tuesday': 2, 'Wednesday': 3,
#     'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7
# }
#
# year = int(input("Year: "))
# month_name = input("Month (January, February,...): ")
# mdate = int(input("Day of month (1-31): "))
# day_name = input("Day of week (Monday,...): ")
# time = int(input("Hour of day (0-23): "))
#
# month_num = month_map[month_name]
# day_num = day_map[day_name]
#
# # -----------------------------------------------------------------------------------------------------------------
# X_input = pd.DataFrame([{
#     'Year': year,
#     'Month_num': month_num,
#     'Mdate': mdate,
#     'Day_num': day_num,
#     'Time': time,
# }])
#
# # scale کردن داده‌ها برای Linear Regression
# X_scaled = scaler.transform(X_input)
#
# # -----------------------------------------------predict------------------------------------------------------------
# X_scaled = scaler.transform(X_input)
# y_pred_linear = np.expm1(linear_model.predict(X_scaled))
# y_predict_rf = rf_model.predict(X_input)
# y_predict_gb = gb_model.predict(X_input)
# Y_predict_xgb = xgb_model.predict(X_input)
#
# X_lstm = np.tile(X_scaled, (48, 1))  # sequence مصنوعی
# X_lstm = np.expand_dims(X_lstm, axis=0)
# Y_predict_lstm = lstm_model.predict(X_lstm)
# Y_predict_lstm_inv = scaler.inverse_transform(
#     np.hstack([Y_predict_lstm, np.zeros((len(Y_predict_lstm), X_scaled.shape[1]-1))])
# )[:, 0]
#
# # -------------------------------------
# print("\nPredicted pedestrian count:")
# print(f"Linear Regression: {y_pred_linear[0]:.0f}")
# print(f"Random Forest: {y_predict_rf[0]:.0f}")
# print(f"Gradient Boosting: {y_predict_gb[0]:.0f}")
# print(f"XGBoost: {Y_predict_xgb[0]:.0f}")
# print(f"LSTM Model: {Y_predict_lstm_inv[0]:.0f}")

import joblib
import numpy as np
import pandas as pd

# ------------------------------------ load model ------------------------------------
model = joblib.load("rf_pipeline.pkl")
print("Random Forest model loaded successfully!")

# ------------------------------------ user input ------------------------------------
year = int(input("Year: "))
month = int(input("Month (1-12): "))
day = int(input("Day of month (1-31): "))
weekday = int(input("Day of week (1=Mon ... 7=Sun): "))
hour = int(input("Hour of day (0-23): "))
sensor_name = input("Sensor name (exactly as in training): ")

# ------------------------------------ feature creation ------------------------------------
# ویژگی‌های سینوسی و کسینوسی برای زمان، ماه، روز
month_sin = np.sin(2 * np.pi * month / 12)
month_cos = np.cos(2 * np.pi * month / 12)
day_sin = np.sin(2 * np.pi * weekday / 7)
day_cos = np.cos(2 * np.pi * weekday / 7)
time_sin = np.sin(2 * np.pi * hour / 24)
time_cos = np.cos(2 * np.pi * hour / 24)

# مقدار lagها چون برای زمان حال وجود ندارند، با صفر پر می‌کنیم
lag_features = {
    'lag_1': 0,
    'lag_2': 0,
    'lag_3': 0,
    'lag_12': 0,
    'lag_24': 0
}

# ساخت DataFrame ورودی نهایی
X_input = pd.DataFrame([{
    'Month_sin': month_sin,
    'Month_cos': month_cos,
    'Day_sin': day_sin,
    'Day_cos': day_cos,
    'Time_sin': time_sin,
    'Time_cos': time_cos,
    'Sensor_Name': sensor_name,
    **lag_features
}])

# ------------------------------------ prediction ------------------------------------
y_pred_log = model.predict(X_input)
y_pred = np.expm1(y_pred_log)

print("Predicted pedestrian count:")
print(f"Random Forest prediction: {y_pred[0]:.0f}")

