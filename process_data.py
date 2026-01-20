#1. IMPORTING LIBRARIES 
#This section loads the necessary tools. We need Pandas to handle data tables, Requests to talk to the API, and Scikit-Learn for the actual machine learning algorithm.

import pandas as pd
import numpy as np
import json
import requests
import time
from sklearn.ensemble import HistGradientBoostingRegressor

#2. CONFIGURATION
# This acts as the "settings menu" for your script. instead of hard-coding URLs later on, we define them here.
# This allows you to scale. If you wanted to add a 4th chart (e.g., "Deaths"), you would just add one line here, and the rest of the script would automatically train a new machine learning model for it without any extra coding.

METRICS = {
    "positivity": {
        "url_suffix": "influenza_testing_positivityByWeek",
        "name": "PCR Positivity Rate (%)"
    },
    "hospital": {
        "url_suffix": "influenza_healthcare_hospitalAdmissionRateByWeek", 
        "name": "Hospital Admission Rate (per 100k)"
    },
    "icu": {
        "url_suffix": "influenza_healthcare_ICUHDUadmissionRateByWeek",
        "name": "ICU/HDU Admission Rate (per 100k)"
    }
}

BASE_URL = "https://api.ukhsa-dashboard.data.gov.uk/themes/infectious_disease/sub_themes/respiratory/topics/Influenza/geography_types/Nation/geographies/England/metrics/"

#3. DATA FETCHING FUNCTION
# Machine learning requires clean, historical data. This function handles the logistics of getting that data from the government API.
# Pagination: The API doesn't send all data at once; it sends it in "pages." The while loop ensures we collect every single page before moving on.
# Data Cleaning: The last few lines convert the text dates (e.g., "2024-01-01") into actual datetime objects and sort them chronologically. ML models strictly require time-series data to be in the correct order.

def fetch_data(metric_suffix):
    url = f"{BASE_URL}{metric_suffix}"
    print(f"  Fetching data from: {metric_suffix}...")
    
    all_data = []
    current_url = f"{url}?page_size=365&format=json"
    
    while current_url:
        try:
            response = requests.get(current_url)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            data = response.json()
            all_data.extend(data['results'])
            current_url = data['next']
            time.sleep(0.2)
        except Exception as e:
            print(f"    Error fetching data: {e}")
            break
            
    if not all_data:
        return None

    df = pd.DataFrame(all_data)
    df = df[['date', 'metric_value']].copy()
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    df.set_index('date', inplace=True)
    return df

#4. FEATURE ENGINEERING (PREPARING THE MODEL)
#This is the start of the core logic. Raw dates (like "2024-01-01") are hard for a model to interpret mathematically. We convert them into Cyclical Features.
# We extract the Week Number (1-52) and Month (1-12). This tells the model that "Week 52" (December) is similar to "Week 1" (January), helping it learn the annual winter peak of influenza.

def train_and_forecast(df):
    # 1. Feature Engineering
    df['Week_Number'] = df.index.isocalendar().week.astype(int)
    df['Month'] = df.index.month
    
    seasonal_features = ['Week_Number', 'Month']
    target = 'metric_value'

#5. TRAINING MODEL 1: THE SEASONAL BASELINE 
# We use a Hybrid Strategy. First, we train a model to learn the general yearly pattern (seasonality).
# The model looks at years of history to understand that flu generally rises in winter and falls in summer.
# We subtract this "General Prediction" from the "Actual Data." The result (Residual) is the "unexpected" part of the flu season—the spikes or drops that are specific to this specific year.
    
    # 2. Train Seasonal Baseline
    model_seasonal = HistGradientBoostingRegressor(categorical_features=[0, 1], random_state=42)
    model_seasonal.fit(df[seasonal_features], df[target])
    
    df['Seasonal_Pred'] = model_seasonal.predict(df[seasonal_features])
    df['Residual'] = df['metric_value'] - df['Seasonal_Pred']

#6. TRAINING MODEL 2: THE RESIDUAL MODEL
# Now we train a second model to predict those specific deviations (Residuals) using Lag Features.
# shift(1): This creates a column representing "Last week's error."
# The Logic: The model learns that if the residuals were high for the last 3 weeks (meaning the flu is spreading faster than the seasonal norm), next week is also likely to be high. This allows the model to react to current trends.

    # 3. Train Residual Model (Lags)
    df['Res_Lag_1'] = df['Residual'].shift(1)
    df['Res_Lag_2'] = df['Residual'].shift(2)
    df['Res_Lag_3'] = df['Residual'].shift(3)
    
    df_resid = df.dropna().copy()
    resid_features = ['Res_Lag_1', 'Res_Lag_2', 'Res_Lag_3', 'Week_Number']
    
    model_resid = HistGradientBoostingRegressor(random_state=42)
    model_resid.fit(df_resid[resid_features], df_resid['Residual'])

# 7. RECURSIVE FORECASTING LOOP
# This is how we predict 52 weeks into the future when we don't have data yet. We use a Sliding Window technique.

    # 4. Generate Future Forecast (52 Weeks)
    last_date = df.index[-1]
    history_residuals = df['Residual'].iloc[-3:].tolist()
    current_date = last_date
    future_forecasts = []

    # BRIDGE: Start forecast from the last actual point to ensure lines connect visually
    future_forecasts.append({
        'date': last_date.strftime('%Y-%m-%d'),
        'Seasonal_Base': float(df['Seasonal_Pred'].iloc[-1]),
        'Final_Forecast': float(df['metric_value'].iloc[-1]) # Start at actual
    })

    for i in range(52):
        current_date = current_date + pd.Timedelta(days=7)
        
        feat_week = current_date.isocalendar().week
        feat_month = current_date.month
        seasonal_base = model_seasonal.predict(pd.DataFrame([[feat_week, feat_month]], columns=seasonal_features))[0]
        
        res_lag_1 = history_residuals[-1]
        res_lag_2 = history_residuals[-2]
        res_lag_3 = history_residuals[-3]
        pred_residual = model_resid.predict(pd.DataFrame([[res_lag_1, res_lag_2, res_lag_3, feat_week]], columns=resid_features))[0]
        
        final_pred = seasonal_base + pred_residual
        final_pred = max(0, final_pred)

        history_residuals.append(pred_residual)
        future_forecasts.append({
            'date': current_date.strftime('%Y-%m-%d'),
            'Seasonal_Base': float(seasonal_base),
            'Final_Forecast': float(final_pred)
        })
        
    return future_forecasts

#8.  MAIN EXECUTION
# This final block orchestrates the whole process.
# Automation: It loops through the dictionary defined at the start, runs the ML pipeline for each one, and dumps the result into a single dashboard_data.json file that your HTML page reads to display the graphs.

full_dashboard_data = {}
print("Starting Multi-Metric Forecast Job...")

for key, config in METRICS.items():
    print(f"\nProcessing: {config['name']}")
    df = fetch_data(config['url_suffix'])
    
    if df is not None and not df.empty:
        forecasts = train_and_forecast(df)
        
        full_dashboard_data[key] = {
            "meta": {"name": config['name']},
            "history": {
                "dates": df.index.strftime('%Y-%m-%d').tolist(),
                "values": df['metric_value'].tolist()
            },
            "forecast": {
                "dates": [x['date'] for x in forecasts],
                "values": [x['Final_Forecast'] for x in forecasts],
                "baseline": [x['Seasonal_Base'] for x in forecasts]
            }
        }
    else:
        print("  Skipping (No data found).")

with open('dashboard_data.json', 'w') as f:
    json.dump(full_dashboard_data, f)

print("\nSuccess: 'dashboard_data.json' updated.")

