import pandas as pd
import numpy as np
import json
import requests
import time
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------
METRICS = {
    # --- INFLUENZA ---
    "positivity": {
        "topic": "Influenza",
        "metric_id": "influenza_testing_positivityByWeek",
        "name": "Flu: PCR Positivity Rate (%)",
        "agg": "mean" 
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

    # --- COVID-19 ---
    "covid_positivity": {
        "topic": "COVID-19",
        "metric_id": "COVID-19_testing_positivity7DayRolling",
        "name": "COVID: PCR Positivity Rate (%)",
        "agg": "mean" 
    },
    "covid_hospital": {
        "topic": "COVID-19",
        "metric_id": "COVID-19_healthcare_admissionByDay", 
        "name": "COVID: Weekly Hospital Admissions",
        "agg": "sum"
    }
}

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
            time.sleep(0.1)
        except Exception as e:
            print(f"    Error fetching data: {e}")
            break
            
    if not all_data:
        return None

    df = pd.DataFrame(all_data)
    df = df[['date', 'metric_value']].copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # Deduplication
    if config['agg'] == 'sum':
        df = df.groupby('date')['metric_value'].sum().reset_index()
    else:
        df = df.groupby('date')['metric_value'].mean().reset_index()

    df.sort_values('date', inplace=True)
    df.set_index('date', inplace=True)
    
    # Weekly Resampling
    if config['agg'] == 'sum':
        df = df.resample('W-SUN')['metric_value'].sum().to_frame()
    else:
        df = df.resample('W-SUN')['metric_value'].mean().to_frame()
    
    df = df[df['metric_value'] > 0.001].copy()
    
    return df

# --- IMPROVED BACKTESTING FUNCTION ---
def evaluate_accuracy(df, weeks_back=52):
    """
    Simulates the past year week-by-week to test accuracy.
    """
    # We need enough history to train (at least 20 weeks) + the backtest period
    if len(df) < (weeks_back + 20):
        print("    Not enough data for full backtest, shortening window...")
        weeks_back = max(1, len(df) - 20)

    dates = []
    actuals = []
    predictions = []

    print(f"    Running backtest on last {weeks_back} weeks...")

    # Loop through the past X weeks
    for i in range(weeks_back, 0, -1):
        # 1. Define the "Current" date in the simulation
        train_end_index = len(df) - i
        
        # 2. Split data: Model only sees data BEFORE this date
        train_df = df.iloc[:train_end_index].copy()
        
        # The target we are trying to predict
        target_date = df.index[train_end_index]
        actual_value = df.iloc[train_end_index]['metric_value']
        
        # 3. Train Model (Identical logic to main forecast)
        train_df['Week_Number'] = train_df.index.isocalendar().week.astype(int)
        train_df['Month'] = train_df.index.month
        
        seasonal_feats = ['Week_Number', 'Month']
        model_seasonal = HistGradientBoostingRegressor(categorical_features=[0, 1], random_state=42)
        model_seasonal.fit(train_df[seasonal_feats], train_df['metric_value'])
        
        train_df['Seasonal_Pred'] = model_seasonal.predict(train_df[seasonal_feats])
        train_df['Residual'] = train_df['metric_value'] - train_df['Seasonal_Pred']
        
        train_df['Res_Lag_1'] = train_df['Residual'].shift(1)
        train_df['Res_Lag_2'] = train_df['Residual'].shift(2)
        train_df['Res_Lag_3'] = train_df['Residual'].shift(3)
        
        df_resid = train_df.dropna()
        if len(df_resid) < 10: continue # Skip if not enough lag data yet

        resid_feats = ['Res_Lag_1', 'Res_Lag_2', 'Res_Lag_3', 'Week_Number']
        model_resid = HistGradientBoostingRegressor(random_state=42)
        model_resid.fit(df_resid[resid_feats], df_resid['Residual'])
        
        # 4. Predict
        feat_week = target_date.isocalendar().week
        feat_month = target_date.month
        last_residuals = train_df['Residual'].iloc[-3:].tolist()
        
        pred_seasonal = model_seasonal.predict(pd.DataFrame([[feat_week, feat_month]], columns=seasonal_feats))[0]
        pred_resid = model_resid.predict(pd.DataFrame([[last_residuals[-1], last_residuals[-2], last_residuals[-3], feat_week]], columns=resid_feats))[0]
        
        final_pred = max(0, pred_seasonal + pred_resid)
        
        dates.append(target_date.strftime('%Y-%m-%d'))
        actuals.append(float(actual_value))
        predictions.append(float(final_pred))

    # Calculate Metrics
    if not actuals: return None
    
    mae = mean_absolute_error(actuals, predictions)
    
    # Calculate Mean Absolute Percentage Error (MAPE)
    # Avoid division by zero
    mape = np.mean(np.abs((np.array(actuals) - np.array(predictions)) / np.maximum(np.array(actuals), 1))) * 100

    return {
        "dates": dates,
        "actuals": actuals,
        "predictions": predictions,
        "mae": float(mae),
        "mape": float(mape)
    }

def train_and_forecast(df):
    df['Week_Number'] = df.index.isocalendar().week.astype(int)
    df['Month'] = df.index.month
    seasonal_features = ['Week_Number', 'Month']
    
    model_seasonal = HistGradientBoostingRegressor(categorical_features=[0, 1], random_state=42)
    model_seasonal.fit(df[seasonal_features], df['metric_value'])
    df['Seasonal_Pred'] = model_seasonal.predict(df[seasonal_features])
    df['Residual'] = df['metric_value'] - df['Seasonal_Pred']

    df['Res_Lag_1'] = df['Residual'].shift(1)
    df['Res_Lag_2'] = df['Residual'].shift(2)
    df['Res_Lag_3'] = df['Residual'].shift(3)
    df_resid = df.dropna().copy()
    resid_features = ['Res_Lag_1', 'Res_Lag_2', 'Res_Lag_3', 'Week_Number']
    
    model_resid = HistGradientBoostingRegressor(random_state=42)
    model_resid.fit(df_resid[resid_features], df_resid['Residual'])

    last_date = df.index[-1]
    history_residuals = df['Residual'].iloc[-3:].tolist()
    current_date = last_date
    future_forecasts = []

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
        final_pred = max(0, seasonal_base + pred_residual)
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
        if len(df) < 20:
             print("  Skipping (Not enough data points).")
             continue
        
        # 1. Generate Future Forecast
        forecasts = train_and_forecast(df)
        
        # 2. Evaluate Past Accuracy (52-Week Backtest)
        # This simulates predicting every single week of the past year.
        accuracy_data = evaluate_accuracy(df, weeks_back=52)

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
            },
            "accuracy": accuracy_data
        }
    else:
        print("  Skipping (No data found).")

with open('dashboard_data.json', 'w') as f:
    json.dump(full_dashboard_data, f)

print("\nSuccess: 'dashboard_data.json' updated.")
