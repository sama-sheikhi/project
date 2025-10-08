import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
scaler = joblib.load("models/scaler.pkl")
linear_model = joblib.load("models/linear_model.pkl")
rf_model = joblib.load("models/random_forest_model.pkl")
gb_model = joblib.load("models/gradient_boosting_model.pkl")
# -------------------------------------------------load models----------------------------------------------------------
scaler = joblib.load("models/scaler.pkl")
linear_model = joblib.load("models/linear_model.pkl")
rf_model = joblib.load("models/random_forest_model.pkl")
gb_model = joblib.load("models/gradient_boosting_model.pkl")
xgb_model = joblib.load("models/xgb_model.pkl")
month_map = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12
}

day_map = {
    'Monday': 1, 'Tuesday': 2, 'Wednesday': 3,
    'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7
}

year = int(input("Year: "))
month_name = input("Month (January, February,...): ")
mdate = int(input("Day of month (1-31): "))
day_name = input("Day of week (Monday,...): ")
time = int(input("Hour of day (0-23): "))

month_num = month_map[month_name]
day_num = day_map[day_name]

# -----------------------------------------------============-----------------------------------------------------------
X_input = pd.DataFrame([{
    'Year': year,
    'Month_num': month_num,
    'Mdate': mdate,
    'Day_num': day_num,
    'Time': time,
}])

# scale کردن داده‌ها برای Linear Regression
X_scaled = scaler.transform(X_input)

# -----------------------------------------------predict------------------------------------------------------------
X_scaled = scaler.transform(X_input)
y_pred_linear = np.expm1(linear_model.predict(X_scaled))

y_pred_linear = np.expm1(linear_model.predict(X_scaled))
y_predict_rf = rf_model.predict(X_input)
y_predict_gb = gb_model.predict(X_input)
Y_predict_xgb = xgb_model.predict(X_input)

# -------------------------------------
print("\nPredicted pedestrian count:")
print(f"Linear Regression: {y_pred_linear[0]:.0f}")
print(f"Random Forest: {y_predict_rf[0]:.0f}")
print(f"Gradient Boosting: {y_predict_gb[0]:.0f}")
print(f"XGBoost: {Y_predict_xgb[0]:.0f}")

