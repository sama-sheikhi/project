import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # فقط خطاهای جدی نمایش داده بشه
import tensorflow as tf

# if os.environ.get("MPLBACKEND", "").startswith("module://"):
#     os.environ["MPLBACKEND"] = "TkAgg"

# matplotlib.use('TkAgg')
# matplotlib.use('macosx')
import scipy.stats as st
from sklearn.model_selection import train_test_split,cross_val_score
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler,MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping

# dt = pd.read_csv("count.csv")
dt = pd.read_csv("sample.csv")
# print(dt.head())
# print(dt.info())
# print(dt.describe())
# print(dt.isnull().sum())
# print("Duplicate rows:", dt.duplicated().sum())
# print(dt.shape)

dt.columns = dt.columns.str.strip()
dt['total_count'] = dt.groupby('Sensor_Name')['Hourly_Counts'].transform('sum')

month_map = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12
}

day_map = {
    'Monday': 1, 'Tuesday': 2, 'Wednesday': 3,
    'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7
}

dt['Month_num'] = dt['Month'].map(month_map)
dt['Day_num'] = dt['Day'].map(day_map)

# ----------------------------------------------------------------------------------------------------------------------
X =dt[['Year','Month_num','Mdate','Day_num','Time']]
Y = dt['Hourly_Counts']
# print(X)
# print(Y)

sample_dt = dt.sample(n=2000, random_state=42)
sample_dt.to_csv("sample.csv", index=False)
# ----------------------------------------------------------------------------------------------------------------------
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42)


quantitative = [f for f in X.columns if X.dtypes[f] != 'object']
qualitative = [f for f in X.columns if X.dtypes[f] == 'object']
print("Quantitative:", quantitative)
print("Qualitative:", qualitative)

# ----------------------------------------------------------------------------------------------------------------------
sns.histplot(dt['Hourly_Counts'], kde=True, bins=30)
plt.title("Histogram of Hourly_Counts")
plt.show()
# ----------------------------------------------------------------------------------------------------------------------
Y = Y_train

plt.figure(2)
plt.title('Johnson SU')
params = st.johnsonsu.fit(Y)
x = np.linspace(Y.min(), Y.max(), 100)
plt.plot(x, st.johnsonsu.pdf(x, *params), color="red", lw=2)
plt.show()

plt.figure(3)
plt.title('Normal')
params = st.norm.fit(Y)
x = np.linspace(Y.min(), Y.max(), 100)
plt.plot(x, st.norm.pdf(x, *params), color="red", lw=2)
plt.show()

plt.figure(4)
plt.title('Log Normal')
params = st.lognorm.fit(Y)
x = np.linspace(Y.min(), Y.max(), 100)
plt.plot(x, st.lognorm.pdf(x, *params), color="red", lw=2)
plt.show()
# ----------------------------------------------------------------------------------------------------------------------
#  p-value < 0.05 → داده نرمال نیست
#  p-value >= 0.05 → داده نرمال هست
test_normality = lambda x: st.shapiro(x.fillna(0))[1] < 0.01
normal = pd.DataFrame(dt[quantitative])
normal = normal.apply(test_normality)
# نتیجه تست هر ستون
print(normal)
# همه ستونها نرمال باشن → True
# حتی یکی غیرنرمال باشه → False
print(not normal.any())
# ----------------------------------------------------------------------------------------------------------------------
# همه ستون‌های عددی رو می‌ریزه توی دو ستون
f = pd.melt(dt, value_vars=quantitative)
# برای هر ستون عددی یک نمودار درست میکنه
g = sns.FacetGrid(f, col="variable", col_wrap=2, sharex=False, sharey=False)
g.map(sns.histplot, "value", kde=True)
plt.show()
# ----------------------------------------------------------------------------------------------------------------------
plt.figure(figsize=(8,5))
sns.boxplot(y=dt['Hourly_Counts'])
plt.title("Boxplot of Hourly_Counts")
plt.show()
# ---------------------------------------------LinearRegression---------------------------------------------------------
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
#
# # ساخت مدل
# model = LinearRegression()
# # تبدیل لگاریتمی روی متغیر هدف
# y_log = np.log1p(Y)
# # آموزش مدل روی داده‌های لگاریتمی
# model.fit(X_train, y_log)
# # پیش‌بینی روی داده‌های تست
# y_pred_log = model.predict(X_test)
# # برگردوندن پیش‌بینی‌ها به مقیاس اصلی
# y_pred = np.expm1(y_pred_log)
# # آموزش مدل
# # model.fit(X_train, Y_train)
# # # پیش‌بینی
# Y_predict = model.predict(X_test)
#
# mae = mean_absolute_error(Y_test, Y_predict)
# rmse = np.sqrt(mean_squared_error(Y_test, Y_predict))
# r2 = r2_score(Y_test, Y_predict)
#
# print(f"MAE: {mae:.2f}")
# print(f"RMSE: {rmse:.2f}")
# print(f"R2: {r2:.2f}")
#
# plt.scatter(Y_test, Y_predict)
# plt.xlabel('Actual')
# plt.ylabel('Predicted')
# plt.title('Actual vs Predicted')
# plt.show()
# ---------------------------------------------RandomForestRegressor----------------------------------------------------
# rf = RandomForestRegressor(random_state=42)
# rf.fit(X_train, Y_train)
# y_predict_rf = rf.predict(X_test)
#
# print('R2 (Random Forest):', r2_score(Y_test, y_predict_rf))
# ---------------------------------------------GradientBoostingRegressor------------------------------------------------
# gb_model = GradientBoostingRegressor(
#     n_estimators=200,
#     learning_rate=0.1,
#     max_depth=3,
#     random_state=42
# )
#
# gb_model.fit(X_train, Y_train)
# Y_predict_gb = gb_model.predict(X_test)
#
# mae_gb = mean_absolute_error(Y_test, Y_predict_gb)
# rmse_gb = np.sqrt(mean_squared_error(Y_test, Y_predict_gb))
# r2_gb = r2_score(Y_test, Y_predict_gb)
# print(f"Gradient Boosting:")
# print(f"MAE: {mae_gb:.2f}")
# print(f"RMSE: {rmse_gb:.2f}")
# print(f"R2: {r2_gb:.2f}")
# ---------------------------------------------------XGBoost------------------------------------------------------------
# xgb_model = xgb.XGBRegressor(
#     n_estimators=200,
#     learning_rate=0.1,
#     max_depth=3,
#     random_state=42
# )
#
# xgb_model.fit(X_train, Y_train)
#
# Y_pred = xgb_model.predict(X_test)
#
# mae = mean_absolute_error(Y_test, Y_pred)
# rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
# r2 = r2_score(Y_test, Y_pred)
#
# print(f"XGBoost:")
# print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.2f}")
# ---------------------------------------------------LSTM---------------------------------------------------------------
# # مرتب‌سازی داده‌ها
# dt = dt.sort_values(by="Date_Time")
# # انتخاب ستون هدف
# data = dt[['Hourly_Counts']].values
# # نرمال‌سازی داده‌ها
# scaler = MinMaxScaler()
# data_scaled = scaler.fit_transform(data)
# # ساخت Sequence
# # از 24 ساعت گذشته برای پیش‌بینی ساعت بعد استفاده می‌کنیم
# X, y = [], []
# n_steps = 24
# for i in range(n_steps, len(data_scaled)):
#     X.append(data_scaled[i-n_steps:i, 0])
#     y.append(data_scaled[i, 0])
# X, y = np.array(X), np.array(y)
# X = X.reshape((X.shape[0], X.shape[1], 1))  # (samples, timesteps, features)
# # ساخت مدل
# model = Sequential()
# model.add(LSTM(64, return_sequences=True, input_shape=(n_steps,1)))
# model.add(LSTM(32))
# model.add(Dense(1))
# model.compile(optimizer='adam', loss='mse')
# # آموزش مدل
# model.fit(X, y, epochs=20, batch_size=16, verbose=1)
# # پیش‌بینی و معکوس کردن نرمال‌سازی
# y_pred = model.predict(X)
# y_pred_inv = scaler.inverse_transform(y_pred)
# y_inv = scaler.inverse_transform(y.reshape(-1,1))
# # ارزیابی
# r2 = r2_score(y_inv, y_pred_inv)
# mae = mean_absolute_error(y_inv, y_pred_inv)
# rmse = np.sqrt(mean_squared_error(y_inv, y_pred_inv))
# print(f"R2: {r2:.3f}, MAE: {mae:.3f}, RMSE: {rmse:.3f}")
# # ذخیره مدل و scaler
# model.save("models/lstm_model.keras")
# joblib.dump(scaler, "models/lstm_scaler.pkl")


df = pd.read_csv("sample.csv")
df.columns = df.columns.str.strip()

# تبدیل Month و Day به عددی
month_map = {'January':1,'February':2,'March':3,'April':4,'May':5,'June':6,
             'July':7,'August':8,'September':9,'October':10,'November':11,'December':12}
day_map = {'Monday':1,'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5,'Saturday':6,'Sunday':7}

df['Month_num'] = df['Month'].map(month_map)
df['Day_num'] = df['Day'].map(day_map)

# مرتب کردن بر اساس زمان
df = df.sort_values(by="Time")

# ویژگی‌های ورودی
features = ['Hourly_Counts', 'Month_num', 'Day_num', 'Time']  # اضافه کردن Hour، Month_num، Day_num
data = df[features].values

# 2️⃣ نرمال‌سازی
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# 3️⃣ ساخت sequence
X, y = [], []
n_steps = 48  # 48 ساعت گذشته برای پیش‌بینی 1 ساعت آینده
for i in range(n_steps, len(data_scaled)):
    X.append(data_scaled[i-n_steps:i])
    y.append(data_scaled[i, 0])  # پیش‌بینی Hourly_Counts

X, y = np.array(X), np.array(y)

# 4️⃣ مدل LSTM
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# 5️⃣ Early stopping برای جلوگیری از overfitting
es = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

# 6️⃣ آموزش مدل
model.fit(X, y, epochs=30, batch_size=16, callbacks=[es], verbose=1)

# 7️⃣ پیش‌بینی
y_pred = model.predict(X)
y_pred_inv = scaler.inverse_transform(np.hstack([y_pred, np.zeros((len(y_pred), data.shape[1]-1))]))[:,0]
y_inv = scaler.inverse_transform(np.hstack([y.reshape(-1,1), np.zeros((len(y), data.shape[1]-1))]))[:,0]

# 8️⃣ ارزیابی
r2 = r2_score(y_inv, y_pred_inv)
mae = mean_absolute_error(y_inv, y_pred_inv)
rmse = np.sqrt(mean_squared_error(y_inv, y_pred_inv))

print(f"R2: {r2:.3f}")
print(f"MAE: {mae:.3f}")
print(f"RMSE: {rmse:.3f}")

model.save("ped_count/lstm_model.keras")
joblib.dump(scaler, "ped_count/lstm_scaler.pkl")
print("Model and scaler saved successfully!")

# -----------------------------------------------make models------------------------------------------------------------
# joblib.dump(scaler, "ped_count/scaler.pkl")
#
# joblib.dump(model, "models/linear_model.pkl")
#
# joblib.dump(rf, "models/random_forest_model.pkl")
#
# joblib.dump(gb_model, "models/gradient_boosting_model.pkl")
#
# joblib.dump(xgb_model, "ped_count/xgb_model.pkl")

