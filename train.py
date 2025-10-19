import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
# ----------------------------------------------------------------------------------------------------------------------
dt = pd.read_csv("sample.csv")
# حذف فاصله‌های اضافی از نام ستون‌ها
dt.columns = dt.columns.str.strip()

dt['Datetime'] = pd.to_datetime(dt['Year'].astype(str) + '-' + dt['Month'].astype(str) + '-' + dt['Mdate'].astype(str) + ' ' + dt['Time'].astype(str) + ':00')
dt = dt.sort_values('Datetime')

# map month/day
month_map = {'January':1, 'February':2, 'March':3, 'April':4, 'May':5, 'June':6, 'July':7, 'August':8, 'September':9, 'October':10, 'November':11, 'December':12}
dt['Month_num'] = dt['Month'].map(month_map)
dt['Day_num'] = dt['Datetime'].dt.dayofweek + 1
# ----------------------------------------------------------------------------------------------------------------------
# cyclical encoding تبدیل چرخه ای
# با استفاده از sin و cos زمان بصورت دایره ای کدگذاری میشه تا مرز های انتهایی مصنوعی نباشه
dt['Time_hour'] = dt['Datetime'].dt.hour
dt['Time_sin'] = np.sin(2 * np.pi * dt['Time_hour'] / 24)
dt['Time_cos'] = np.cos(2 * np.pi * dt['Time_hour'] / 24)
dt['Month_sin'] = np.sin(2 * np.pi * dt['Month_num'] / 12)
dt['Month_cos'] = np.cos(2 * np.pi * dt['Month_num'] / 12)
dt['Day_sin'] = np.sin(2 * np.pi * dt['Day_num'] / 7)
dt['Day_cos'] = np.cos(2 * np.pi * dt['Day_num'] / 7)

# ----------------------------------------------------------------------------------------------------------------------
# حذف داده‌های پرت برای هر حسگر جداگانه
# Median Absolute Deviation = (MAD) نوعی مقدار پراکندگی مقاوم
def clip_outliers(group):
    # mad → نشان‌دهنده‌ی پراکندگی داده‌ها حول میانه است
    # group.median() → میانه‌ی داده‌های آن گروه
    # np.abs(group - group.median()) → قدرمطلق فاصله‌ی هر مقدار از میانه
    # np.median(...) → میانه ی این فاصله ها
    # *1.4826 → ضریب تبدیل MADبه واحدی معادل انحراف معیار
    mad = np.median(np.abs(group - group.median())) * 1.4826
    # در اینجا بازه تعیین مینیم
    # هر مقداری کمتر از lower پرت پایین است
    lower = group.median() - 2 * mad
    # هر قداری بیشتر از upper پرت بالا است
    upper = group.median() + 2 * mad
    # درنهایت این تابع باعث میشود اگر مقداری کمتر از lower باشد باید مقدارش برابر lower شود و برعکس برای upper
    # یعنی داده های پرت بریده میشوند نه حذف!
    return group.clip(lower=lower, upper=upper)

dt['Hourly_Counts'] = dt.groupby('Sensor_Name')['Hourly_Counts'].transform(clip_outliers)

# ----------------------------------------------------------------------------------------------------------------------
# تبدیل لگاریتمی برای کاهش تأثیر مقدارهای خیلی بزرگ
dt['Hourly_Counts_log'] = np.log1p(dt['Hourly_Counts'])

# ----------------------------------------------------------------------------------------------------------------------
# ساخت ویژگی‌های تاخیری (Lag Features)
# یعنی برای هر ردیف، مقادیر ساعت‌های قبل از همان حسگر را به‌عنوان ویژگی‌های جدید ذخیره می‌کند
# حافظه زمانی
#  ۱ ۲ ۳ ۱۲ ۲۴ ساعت قبل = لیستی از تأخیرهای زمانی داریم
lags = [1, 2, 3, 12, 24]
for lag in lags:
    dt[f'lag_{lag}'] = dt.groupby('Sensor_Name')['Hourly_Counts_log'].shift(lag)
# وقتی چند lag میسازیم بعضضی سلول ها NaN میشن مثل ردیف هاب اول که مقدار قبلی ندارن و اونارو حذف میکنیم
dt = dt.dropna()

# ----------------------------------------------------------------------------------------------------------------------
features = ['Month_sin', 'Month_cos', 'Day_sin', 'Day_cos', 'Time_sin', 'Time_cos', 'Sensor_Name'] + [f'lag_{lag}' for lag in lags]
X = dt[features]
y = dt['Hourly_Counts_log']

# پیش پردازش
# ویژگی‌های عددی → نرمال‌سازی بین 0 و 1
categorical = ['Sensor_Name']
numerical = [col for col in features if col not in categorical]
# ویژگی متنی Sensor_Name → One-Hot Encoding (ستون‌های باینری)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), numerical),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical)
        # ستون جدید واسه حسگر جدید اضافه نمیکنه و ۰ میداره واسه جلوگیری از خطا
    ])

# ----------------------------------------------------------------------------------------------------------------------
# split chronological تقسیم زمانی داده‌ها
# چون داده‌ها زمانی‌اند، از train_test_split استفاده نمی‌کنیم (تا ترتیب زمان حفظ شود)
# ۸۰٪ آموزش ۲۰٪ تست
train_size = int(len(dt) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# print(f"Train size: {len(X_train)}")

# ----------------------------------------------------------------------------------------------------------------------
model = RandomForestRegressor(random_state=42)

# Pipeline = پیش‌پردازش + مدل در یک زنجیره واحد تا هیچ اطلاعاتی از داده های تست وارد ترین نشن (برای راحتی و جلوگیری از data leakage)
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

# برو داخل pipeline بهش model و چیزهایی که مشخص شده را در مقادیر مختلف تست کن
param_grid = {
    # تعداد درخت
    'model__n_estimators': [100, 200, 300],
    # عمق ماکسیمم هر درخت
    'model__max_depth': [10, 20, None],
    # حذاقل نمونه برای تقسیم گره
    'model__min_samples_split': [2, 5]
}

# ----------------------------------------------------------------------------------------------------------------------
# جستجوی شبکه ای برای تنظیم پارامترها grid search
# یافتن بهترین تنظیمات برای مدل یعنی مقادیری که دقت را بیشینه و خطا را کمینه میکند

# TimeSeriesSplit تقسیم داده به چند بخش زمانی (نه تصادفی)
tscv = TimeSeriesSplit(n_splits=3)

# GridSearchCV تمام ترکیب‌های ممکن از پارامترها را تست می‌کند و بهترین را پیدا می‌کند
# برای هر ترکیب از پارمتر ها پایپ لاین میسازه
# مدل را چندبار اموزش و ارزیابی میکنه
# نوع cross validation یعنی داده هارو چندبار به شکل های مختلف به تس و ترین تقسیم کنیم تا مطمبن بشیم مدل در کل داده ها خوب یادگرفته
# scoring='neg_mean_squared_error': معیار ارزیابی بر اساس کمترین MSE
# نمایش میزان پیشرفت در ترمینال
# n_jobs=-1: استفاده از تمام CPUها برای سرعت بیشتر
grid_search = GridSearchCV(pipeline, param_grid, cv=tscv, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)

# کل فرایند اموزش و ارزیابی رو انجام میده و بهترین مدل رو در داخل خودش ذخیره میکنه
grid_search.fit(X_train, y_train)

# ----------------------------------------------------------------------------------------------------------------------
# prediction
y_pred = grid_search.predict(X_test)

# inverse log
y_test_inv = np.expm1(y_test)
y_pred_inv = np.expm1(y_pred)

# metrics
r2 = r2_score(y_test_inv, y_pred_inv)
# خطای میانگین مطلق
mae = mean_absolute_error(y_test_inv, y_pred_inv)
# خطای میانگین مربعی (حساس‌تر به خطاهای بزرگ)
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
print(f"R2: {r2:.3f}")
print(f"MAE: {mae:.3f}")
print(f"RMSE: {rmse:.3f}")

# ذخیره  مدل
rf_model = grid_search.best_estimator_
joblib.dump(rf_model, "rf_pipeline.pkl")
print("Model saved successfully!")