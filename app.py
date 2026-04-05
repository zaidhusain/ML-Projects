import streamlit as st
import joblib
import pandas as pd
import requests
from datetime import datetime

# Loading trained model
model = joblib.load('xgboost_aqi_model.pkl')
history = pd.read_csv('data/AQI(With Features).csv', index_col=['datetime'])

# Fetching Real-Time Data through Open-Meteo
def get_current_data():
    # Air quality
    aqi_url = (
        "https://air-quality-api.open-meteo.com/v1/air-quality"
        "?latitude=28.6469&longitude=77.3168"
        "&current=us_aqi,pm2_5,pm10,carbon_monoxide,nitrogen_dioxide"
    )

    # Weather
    weather_url = (
        "https://api.open-meteo.com/v1/forecast"
        "?latitude=28.6469&longitude=77.3168"
        "&current=temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m"
        "&wind_speed_unit=kmh"
    )

    aqi_r     = requests.get(aqi_url).json()
    weather_r = requests.get(weather_url).json()

    return {
        'datetime':      datetime.now().replace(minute=0, second=0, microsecond=0),
        'temp_c':        weather_r['current']['temperature_2m'],
        'humidity':      weather_r['current']['relative_humidity_2m'],
        'pressure_mb':   weather_r['current']['pressure_msl'],
        'windspeed_kph': weather_r['current']['wind_speed_10m'],
        'aqi_index':     aqi_r['current']['us_aqi'],
        'pm2_5':         aqi_r['current']['pm2_5'],
        'pm10':          aqi_r['current']['pm10'],
        'co':            aqi_r['current']['carbon_monoxide'],
        'no2':           aqi_r['current']['nitrogen_dioxide'],
    }
new_df = get_current_data()
df_now = pd.DataFrame([new_df]).set_index('datetime')

df = pd.concat([history, df_now])
df.index = pd.to_datetime(df.index)

def feature_engg(df=df):
    # Time Features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.day_of_week
    df['month'] = df.index.month
    df['is_weekend'] = (df.index.day_of_week > 5).astype(int)

    df['season']      = df['month'].map({
        12:0, 1:0, 2:0,    # winter
        3:1, 4:1, 5:1,    # spring
        6:2, 7:2, 8:2,    # summer
        9:3,10:3,11:3     # autumn
    })

    df['is_rush_hour'] = df['hour'].isin([7,8,9,17,18,19]).astype(int)

    # AQI Lags

    df['aqi_lag_1h'] = df['aqi_index'].shift(1)
    df['aqi_lag_24h'] = df['aqi_index'].shift(24)
    df['aqi_lag_168h'] = df['aqi_index'].shift(168) # Last week same hour

    # pollutants lags

    df['pm2_5_lag_24h'] = df['pm2_5'].shift(24)
    df['pm10_lag_24h'] = df['pm10'].shift(24)
    df['co_lag_24h'] = df['co'].shift(24)
    df['no2_lag_24h'] = df['no2'].shift(24)

    # Other LAgs

    df['wind_lag_1h'] = df['windspeed_kph'].shift(1)
    df['humidity_lag_1h'] = df['humidity'].shift(1)
    df['temp_lag_1h'] = df['temp_c'].shift(1)

    # High wind disperses pollution
    df['wind_aqi'] =  df['wind_lag_1h'] * df['aqi_lag_1h']

    # MA and std
    df['aqi_MA_24h'] = df['aqi_index'].rolling(window=24).mean().shift(1)
    df['aqi_MA_168h'] = df['aqi_index'].rolling(window=168).mean().shift(1)

    df['aqi_std_24h'] = df['aqi_index'].rolling(window=24).std().shift(1) # Volatility

    return df

df = feature_engg(df)

def get_safety(aqi):
    if aqi <= 50:
        return "✅ Good", "Safe for everyone including senior citizens", "green"
    elif aqi <= 100:
        return "🟡 Moderate", "Generally safe — sensitive individuals should take care", "orange"
    elif aqi <= 150:
        return "🟠 Unhealthy for Sensitive Groups", "Senior citizens should limit outdoor time", "orange"
    elif aqi <= 200:
        return "🔴 Unhealthy", "NOT safe for senior citizens — stay indoors", "red"
    elif aqi <= 300:
        return "🟣 Very Unhealthy", "Hazardous — everyone should avoid going out", "red"
    else:
        return "⚫ Hazardous", "Emergency conditions — do not go outside", "red"
 
predictors = [
    'hour', 'day_of_week', 'month', 'is_weekend', 'season', 'is_rush_hour',
    'aqi_lag_1h', 'aqi_lag_24h', 'aqi_lag_168h',
    'pm2_5_lag_24h', 'pm10_lag_24h', 'co_lag_24h', 'no2_lag_24h',
    'wind_lag_1h', 'humidity_lag_1h', 'temp_lag_1h',
    'wind_aqi',
    'aqi_MA_24h', 'aqi_MA_168h', 'aqi_std_24h'
]


latest = df[predictors].iloc[[-1]]
predicted_aqi = round(float(model.predict(latest)[0]))
current = new_df   # raw current hour data

label, advice, color = get_safety(predicted_aqi)

# Streamlit UI

st.set_page_config(page_title="Delhi AQI Safety", page_icon="🌫️", layout="centered")

st.title("🌫️ Delhi AQI — Is It Safe to Go Out?")
st.caption("Real time air quality prediction for Anand Vihar, Delhi")

st.divider()

# Current Conditions
st.subheader("📍 Current Conditions")
col1, col2, col3, col4 = st.columns(4)
col1.metric("AQI Now",     current['aqi_index'])
col2.metric("Temperature", f"{current['temp_c']}°C")
col3.metric("Humidity",    f"{current['humidity']}%")
col4.metric("Wind Speed",  f"{current['windspeed_kph']}km/h")

st.divider()

# PRediction

st.subheader("Next Hour Prediction")
st.metric("Predicted AQI", predicted_aqi)

if color == "green":
    st.success(f"{label} — {advice}")
elif color == "orange":
    st.warning(f"{label} — {advice}")
else:
    st.error(f"{label} — {advice}")

st.divider()

# Last 24hr Trend
st.subheader("📈 Last 24 Hours AQI Trend")
st.line_chart(df['aqi_index'].tail(24))

# Saving updated history
df.to_csv('data/AQI(With Features).csv')

# Footer
st.divider()
st.caption("Built with XGBoost · Open-Meteo · Streamlit | Anand Vihar, Delhi")
st.caption("~ Zaid Khan")
