import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scipy.stats as st
from sklearn.model_selection import train_test_split,cross_val_score
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import xgboost as xgb

# dt = pd.read_csv("count.csv")
dt = pd.read_csv("sample.csv")
# print(dt.head())
# print(dt.info())
# print(dt.describe())
print(dt.isnull().sum())
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
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ساخت مدل
model = LinearRegression()
# تبدیل لگاریتمی روی متغیر هدف
y_log = np.log1p(Y)
# آموزش مدل روی داده‌های لگاریتمی
model.fit(X_train, y_log)
# پیش‌بینی روی داده‌های تست
y_pred_log = model.predict(X_test)
# برگردوندن پیش‌بینی‌ها به مقیاس اصلی
y_pred = np.expm1(y_pred_log)
# آموزش مدل
# model.fit(X_train, Y_train)
# # پیش‌بینی
Y_predict = model.predict(X_test)

mae = mean_absolute_error(Y_test, Y_predict)
rmse = np.sqrt(mean_squared_error(Y_test, Y_predict))
r2 = r2_score(Y_test, Y_predict)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2: {r2:.2f}")

plt.scatter(Y_test, Y_predict)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
plt.show()
# ---------------------------------------------RandomForestRegressor----------------------------------------------------
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, Y_train)
y_predict_rf = rf.predict(X_test)

print('R2 (Random Forest):', r2_score(Y_test, y_predict_rf))
# ---------------------------------------------GradientBoostingRegressor------------------------------------------------
gb_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

gb_model.fit(X_train, Y_train)
Y_predict_gb = gb_model.predict(X_test)

mae_gb = mean_absolute_error(Y_test, Y_predict_gb)
rmse_gb = np.sqrt(mean_squared_error(Y_test, Y_predict_gb))
r2_gb = r2_score(Y_test, Y_predict_gb)
print(f"Gradient Boosting:")
print(f"MAE: {mae_gb:.2f}")
print(f"RMSE: {rmse_gb:.2f}")
print(f"R2: {r2_gb:.2f}")
# ---------------------------------------------------XGBoost------------------------------------------------------------
xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

xgb_model.fit(X_train, Y_train)

Y_pred = xgb_model.predict(X_test)

mae = mean_absolute_error(Y_test, Y_pred)
rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
r2 = r2_score(Y_test, Y_pred)

print(f"XGBoost:")
print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.2f}")

# -----------------------------------------------make models------------------------------------------------------------
joblib.dump(scaler, "models/scaler.pkl")

joblib.dump(model, "models/linear_model.pkl")

joblib.dump(rf, "models/random_forest_model.pkl")

joblib.dump(gb_model, "models/gradient_boosting_model.pkl")

joblib.dump(xgb_model, "models/xgb_model.pkl")