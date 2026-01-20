import pandas as pd
import numpy as np
import json
import requests
import time
from sklearn.ensemble import HistGradientBoostingRegressor

# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------
METRICS = {
    # --- INFLUENZA (Weekly Data) ---
    "positivity": {
        "topic": "Influenza",
        "metric_id": "influenza_testing_positivityByWeek",
        "name": "Flu: PCR Positivity Rate (%)",
        "agg": "mean" # Already weekly, but good practice
    },
    "hospital": {
        "topic": "Influenza",
        "metric_id": "influenza_healthcare_hospitalAdmissionRateByWeek", 
        "name": "Flu: Hospital Admission Rate",
        "agg": "mean"
    },
    "icu": {
        "topic": "Influenza",
        "metric_id": "influenza_healthcare_ICUHDUadmissionRateByWeek",
        "name": "Flu: ICU/HDU Admission Rate",
        "agg": "mean"
    },

    # --- COVID-19 (Daily Data -> Needs Weekly Aggregation) ---
    "covid_positivity": {
        "topic": "COVID-19",
        "metric_id": "COVID-19_testing_positivity7DayRolling",
        "name": "COVID: PCR Positivity Rate (%)",
        "agg": "mean" # Average the daily rates for the week
    },
    "covid_hospital": {
        "topic": "COVID-19",
        # FIXED: Uses the daily admission count
        "metric_id": "COVID-19_healthcare_admissionByDay",
        "name": "COVID: Weekly Hospital Admissions",
        "agg": "sum" # Sum up daily admissions to get weekly total
    },
    "covid_deaths": {
        "topic": "COVID-19",
        # FIXED: Uses the daily ONS death count
        "metric_id": "COVID-19_deaths_ONSByDay",
        "name": "COVID: Weekly Deaths (ONS)",
        "agg": "sum" # Sum up daily deaths to get weekly total
    }
}

# The URL root.
API_TEMPLATE = "https://api.ukhsa-dashboard.data.gov.uk/themes/infectious_disease/sub_themes/respiratory/topics/{topic}/geography_types/Nation/geographies/England/metrics/{metric_id}"

def fetch_data(config):
    url = API_TEMPLATE.format(topic=config['topic'], metric_id=config['metric_id'])
    print(f"  Fetching {config['topic']} data: {config['metric_id']}...")
    
    all_data = []
    current_url = f"{url}?page_size=365&format=json"
    
    while current_url:
        try:
            response = requests.get(current_url)
            if response.status_code == 404:
                print(f"    Warning: 404 Not Found for {config['metric_id']}")
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
    
    # --- RESAMPLING LOGIC (New) ---
    # Ensure all data is Weekly (Ending on Sunday) to match the ML model structure
    # This handles the mix of Daily (COVID) and Weekly (Flu) data seamlessly.
    original_rows = len(df)
    if config['agg'] == 'sum':
        df = df.resample('W-SUN')['metric_value'].sum().to_frame()
    else:
        df = df.resample('W-SUN')['metric_value'].mean().to_frame()
    
    # Remove any weeks with 0 data that might be artifacts of resampling empty future dates
    df = df[df['metric_value'] > 0].copy()
    
    print(f"    processed {original_rows} raw rows into {len(df)} weekly points.")
    return df

def train_and_forecast(df):
    # 1. Feature Engineering
    df['Week_Number'] = df.index.isocalendar().week.astype(int)
    df['Month'] = df.index.month
    
    seasonal_features = ['Week_Number', 'Month']
    target = 'metric_value'

    # 2. Train Seasonal Baseline
    model_seasonal = HistGradientBoostingRegressor(categorical_features=[0, 1], random_state=42)
    model_seasonal.fit(df[seasonal_features], df[target])
    
    df['Seasonal_Pred'] = model_seasonal.predict(df[seasonal_features])
    df['Residual'] = df['metric_value'] - df['Seasonal_Pred']

    # 3. Train Residual Model (Lags)
    df['Res_Lag_1'] = df['Residual'].shift(1)
    df['Res_Lag_2'] = df['Residual'].shift(2)
    df['Res_Lag_3'] = df['Residual'].shift(3)
    
    df_resid = df.dropna().copy()
    resid_features = ['Res_Lag_1', 'Res_Lag_2', 'Res_Lag_3', 'Week_Number']
    
    model_resid = HistGradientBoostingRegressor(random_state=42)
    model_resid.fit(df_resid[resid_features], df_resid['Residual'])

    # 4. Generate Future Forecast (52 Weeks)
    last_date = df.index[-1]
    history_residuals = df['Residual'].iloc[-3:].tolist()
    current_date = last_date
    future_forecasts = []

    # BRIDGE: Start forecast from the last actual point
    future_forecasts.append({
        'date': last_date.strftime('%Y-%m-%d'),
        'Seasonal_Base': float(df['Seasonal_Pred'].iloc[-1]),
        'Final_Forecast': float(df['metric_value'].iloc[-1])
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

# -------------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------------
full_dashboard_data = {}
print("Starting Multi-Virus Forecast Job...")

for key, config in METRICS.items():
    print(f"\nProcessing: {config['name']}")
    df = fetch_data(config)
    
    if df is not None and not df.empty:
        # Handle cases where data might be too short for ML (e.g. beginning of a dataset)
        if len(df) < 10:
             print("  Skipping (Not enough data points for ML).")
             continue

        forecasts = train_and_forecast(df)
        
        full_dashboard_data[key] = {
            "meta": {"name": config['name'], "topic": config['topic']},
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

print("\nSuccess: 'dashboard_data.json' updated with Flu and COVID data.")
