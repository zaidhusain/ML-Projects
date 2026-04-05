import pandas as pd
import numpy as np
import requests
import joblib
from datetime import datetime, timedelta
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# CONFIG
HISTORY_PATH = 'data/AQI(With Features).csv'
MODEL_PATH   = 'xgboost_aqi_model.pkl'
LAT          = 28.6469
LON          = 77.3168

PREDICTORS = [
    'hour', 'day_of_week', 'month', 'is_weekend', 'season', 'is_rush_hour',
    'aqi_lag_1h', 'aqi_lag_24h', 'aqi_lag_168h',
    'pm2_5_lag_24h', 'pm10_lag_24h', 'co_lag_24h', 'no2_lag_24h',
    'wind_lag_1h', 'humidity_lag_1h', 'temp_lag_1h',
    'wind_aqi',
    'aqi_MA_24h', 'aqi_MA_168h', 'aqi_std_24h'
]

# FETCH LATEST DATA FROM OPEN-METEO

def fetch_aqi_data(start_date, end_date):
    url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude={LAT}&longitude={LON}"
        f"&hourly=pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,us_aqi"
        f"&start_date={start_date}&end_date={end_date}"
        f"&timezone=Asia/Kolkata"
    )
    r = requests.get(url).json()
    hourly = r['hourly']

    df = pd.DataFrame({
        'datetime':  pd.to_datetime(hourly['time']),
        'aqi_index': hourly['us_aqi'],
        'pm2_5':     hourly['pm2_5'],
        'pm10':      hourly['pm10'],
        'co':        hourly['carbon_monoxide'],
        'no2':       hourly['nitrogen_dioxide'],
    })
    return df.set_index('datetime')


def fetch_weather_data(start_date, end_date):
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={LAT}&longitude={LON}"
        f"&hourly=temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m"
        f"&start_date={start_date}&end_date={end_date}"
        f"&timezone=Asia/Kolkata"
        f"&wind_speed_unit=kmh"
    )
    r = requests.get(url).json()
    hourly = r['hourly']

    df = pd.DataFrame({
        'datetime':      pd.to_datetime(hourly['time']),
        'temp_c':        hourly['temperature_2m'],
        'humidity':      hourly['relative_humidity_2m'],
        'pressure_mb':   hourly['pressure_msl'],
        'windspeed_kph': hourly['wind_speed_10m'],
    })
    return df.set_index('datetime')


def fetch_latest_data():
    # Fetch last 40 days to ensure enough history for lag_168h
    end   = datetime.now().strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=40)).strftime('%Y-%m-%d')

    print(f"Fetching data from {start} to {end}...")

    aqi_df     = fetch_aqi_data(start, end)
    weather_df = fetch_weather_data(start, end)

    combined = pd.concat([aqi_df, weather_df], axis=1)
    print(f"Fetched {len(combined)} new rows")
    return combined

# FEATURE ENGINEERING
def feature_engg(df):
    # Time features
    df['hour']        = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month']       = df.index.month
    df['is_weekend']  = (df.index.dayofweek >= 5).astype(int)
    df['season']      = df['month'].map({
        12:0, 1:0, 2:0,
         3:1, 4:1, 5:1,
         6:2, 7:2, 8:2,
         9:3,10:3,11:3
    })
    df['is_rush_hour'] = df.index.hour.isin([7,8,9,17,18,19]).astype(int)

    # AQI lags
    df['aqi_lag_1h']   = df['aqi_index'].shift(1)
    df['aqi_lag_24h']  = df['aqi_index'].shift(24)
    df['aqi_lag_168h'] = df['aqi_index'].shift(168)

    # Pollutant lags
    df['pm2_5_lag_24h'] = df['pm2_5'].shift(24)
    df['pm10_lag_24h']  = df['pm10'].shift(24)
    df['co_lag_24h']    = df['co'].shift(24)
    df['no2_lag_24h']   = df['no2'].shift(24)

    # Weather lags
    df['temp_lag_1h']     = df['temp_c'].shift(1)
    df['humidity_lag_1h'] = df['humidity'].shift(1)
    df['wind_lag_1h']     = df['windspeed_kph'].shift(1)
    df['wind_aqi']        = df['wind_lag_1h'] * df['aqi_lag_1h']

    # Rolling features
    df['aqi_MA_24h']  = df['aqi_index'].rolling(24).mean().shift(1)
    df['aqi_MA_168h'] = df['aqi_index'].rolling(168).mean().shift(1)
    df['aqi_std_24h'] = df['aqi_index'].rolling(24).std().shift(1)

    # Target
    df['target_aqi'] = df['aqi_index'].shift(-1)

    return df

# RETRAIN
def retrain():
    print("=" * 40)
    print(f"Starting retrain — {datetime.now()}")
    print("=" * 40)

    # Load existing history
    print("\nLoading existing history...")
    history = pd.read_csv(HISTORY_PATH, index_col='datetime', parse_dates=True)

    # Keep only raw columns before appending
    raw_cols = ['temp_c', 'humidity', 'pressure_mb', 'windspeed_kph',
                'aqi_index', 'pm2_5', 'pm10', 'co', 'no2']
    history = history[raw_cols]
    print(f"Existing rows: {len(history)}")

    # Fetch latest data
    print("\nFetching latest data from Open-Meteo...")
    new_data = fetch_latest_data()

    # Merge and deduplicate
    print("\nMerging datasets...")
    combined = pd.concat([history, new_data])
    combined = combined[~combined.index.duplicated(keep='last')]
    combined = combined.sort_index()
    print(f"Total rows after merge: {len(combined)}")
    print(f"Date range: {combined.index[0]} to {combined.index[-1]}")

    # Feature engineering
    print("\nBuilding features...")
    combined = feature_engg(combined)
    combined = combined.dropna()
    print(f"Rows after dropna: {len(combined)}")

    # Train/test split
    split   = int(len(combined) * 0.8)
    X_train = combined[PREDICTORS][:split]
    X_test  = combined[PREDICTORS][split:]
    y_train = combined['target_aqi'][:split]
    y_test  = combined['target_aqi'][split:]

    # Retrain model
    print("\nRetraining XGBoost...")
    model = XGBRegressor(n_estimators = 1000, learning_rate = 0.1, gamma= 0.1, max_depth=6, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mae    = mean_absolute_error(y_test, y_pred)
    mape   = mean_absolute_percentage_error(y_test, y_pred) * 100

    print(f"\nRetrain Complete!")
    print(f"MAE:  {mae:.3f}")
    print(f"MAPE: {mape:.4f}%")

    # Save updated model and history
    joblib.dump(model, MODEL_PATH)
    combined.to_csv(HISTORY_PATH)

    print(f"\nModel saved to {MODEL_PATH}")
    print(f"History saved to {HISTORY_PATH}")
    print(f"Retrain finished — {datetime.now()}")
    print("=" * 50)


if __name__ == '__main__':
    retrain()