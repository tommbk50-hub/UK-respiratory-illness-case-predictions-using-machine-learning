import pandas as pd
import numpy as np
import json
import requests
import time
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.inspection import permutation_importance

# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------
METRICS = {
    "positivity": {
        "topic": "Influenza",
        "metric_id": "influenza_testing_positivityByWeek",
        "agg": "mean",
        "name": "Flu: PCR Positivity Rate (%)"
    },
    "hospital": {
        "topic": "Influenza",
        "metric_id": "influenza_healthcare_hospitalAdmissionRateByWeek", 
        "agg": "mean",
        "name": "Flu: Hospital Admission Rate"
    },
    "icu": {
        "topic": "Influenza",
        "metric_id": "influenza_healthcare_ICUHDUadmissionRateByWeek",
        "agg": "mean",
        "name": "Flu: ICU/HDU Admission Rate"
    },
    "covid_positivity": {
        "topic": "COVID-19",
        "metric_id": "COVID-19_testing_positivity7DayRolling",
        "agg": "mean",
        "name": "COVID: PCR Positivity Rate (%)"
    },
    "covid_hospital": {
        "topic": "COVID-19",
        "metric_id": "COVID-19_healthcare_admissionByDay", 
        "agg": "sum",
        "name": "COVID: Weekly Hospital Admissions"
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
    
    if config['agg'] == 'sum':
        df = df.groupby('date')['metric_value'].sum().reset_index()
    else:
        df = df.groupby('date')['metric_value'].mean().reset_index()

    df.sort_values('date', inplace=True)
    df.set_index('date', inplace=True)
    
    if config['agg'] == 'sum':
        df = df.resample('W-SUN')['metric_value'].sum().to_frame()
    else:
        df = df.resample('W-SUN')['metric_value'].mean().to_frame()
    
    df = df[df['metric_value'] > 0.001].copy()
    
    return df

def evaluate_accuracy(df, weeks_back=52):
    if len(df) < (weeks_back + 20):
        weeks_back = max(1, len(df) - 20)

    dates = []
    actuals = []
    predictions = []
    residuals = [] # NEW: Store individual errors

    for i in range(weeks_back, 0, -1):
        train_end_index = len(df) - i
        train_df = df.iloc[:train_end_index].copy()
        
        target_date = df.index[train_end_index]
        actual_value = df.iloc[train_end_index]['metric_value']
        
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
        if len(df_resid) < 10: continue

        resid_feats = ['Res_Lag_1', 'Res_Lag_2', 'Res_Lag_3', 'Week_Number']
        model_resid = HistGradientBoostingRegressor(random_state=42)
        model_resid.fit(df_resid[resid_feats], df_resid['Residual'])
        
        feat_week = target_date.isocalendar().week
        feat_month = target_date.month
        last_residuals = train_df['Residual'].iloc[-3:].tolist()
        
        pred_seasonal = model_seasonal.predict(pd.DataFrame([[feat_week, feat_month]], columns=seasonal_feats))[0]
        pred_resid = model_resid.predict(pd.DataFrame([[last_residuals[-1], last_residuals[-2], last_residuals[-3], feat_week]], columns=resid_feats))[0]
        
        final_pred = max(0, pred_seasonal + pred_resid)
        
        dates.append(target_date.strftime('%Y-%m-%d'))
        actuals.append(float(actual_value))
        predictions.append(float(final_pred))
        
        # Calculate Residual (Error)
        residuals.append(float(actual_value - final_pred))

    if not actuals: return None
    
    mae = mean_absolute_error(actuals, predictions)
    mape = np.mean(np.abs((np.array(actuals) - np.array(predictions)) / np.maximum(np.array(actuals), 1))) * 100

    return {
        "dates": dates,
        "actuals": actuals,
        "predictions": predictions,
        "residuals": residuals, # NEW: List of errors for histogram
        "mae": float(mae),
        "mape": float(mape)
    }

def train_and_forecast(df):
    df['Week_Number'] = df.index.isocalendar().week.astype(int)
    df['Month'] = df.index.month
    seasonal_features = ['Week_Number', 'Month']
    
    model_seasonal = HistGradientBoostingRegressor(categorical_features=[0, 1], loss='squared_error', random_state=42)
    model_seasonal.fit(df[seasonal_features], df['metric_value'])
    df['Seasonal_Pred'] = model_seasonal.predict(df[seasonal_features])
    df['Residual'] = df['metric_value'] - df['Seasonal_Pred']

    df['Res_Lag_1'] = df['Residual'].shift(1)
    df['Res_Lag_2'] = df['Residual'].shift(2)
    df['Res_Lag_3'] = df['Residual'].shift(3)
    df_resid = df.dropna().copy()
    resid_features = ['Res_Lag_1', 'Res_Lag_2', 'Res_Lag_3', 'Week_Number']
    
    model_resid = HistGradientBoostingRegressor(loss='squared_error', random_state=42)
    model_resid.fit(df_resid[resid_features], df_resid['Residual'])

    # Feature Importance
    perm_importance = permutation_importance(model_resid, df_resid[resid_features], df_resid['Residual'], n_repeats=10, random_state=42)
    feature_names = ['Last Week (Lag 1)', '2 Weeks Ago (Lag 2)', '3 Weeks Ago (Lag 3)', 'Seasonal Offset (Week)']
    importance_scores = perm_importance.importances_mean.tolist()
    importance_list = [{"name": n, "score": s} for n, s in zip(feature_names, importance_scores)]
    importance_list.sort(key=lambda x: x['score'], reverse=True)

    model_resid_upper = HistGradientBoostingRegressor(loss='quantile', quantile=0.95, random_state=42)
    model_resid_upper.fit(df_resid[resid_features], df_resid['Residual'])

    model_resid_lower = HistGradientBoostingRegressor(loss='quantile', quantile=0.05, random_state=42)
    model_resid_lower.fit(df_resid[resid_features], df_resid['Residual'])

    last_date = df.index[-1]
    history_residuals = df['Residual'].iloc[-3:].tolist()
    current_date = last_date
    future_forecasts = []

    bridge_val = float(df['metric_value'].iloc[-1])
    future_forecasts.append({
        'date': last_date.strftime('%Y-%m-%d'),
        'Seasonal_Base': float(df['Seasonal_Pred'].iloc[-1]),
        'Final_Forecast': bridge_val,
        'Lower_Bound': bridge_val,
        'Upper_Bound': bridge_val
    })

    for i in range(52):
        current_date = current_date + pd.Timedelta(days=7)
        feat_week = current_date.isocalendar().week
        feat_month = current_date.month
        
        seasonal_base = model_seasonal.predict(pd.DataFrame([[feat_week, feat_month]], columns=seasonal_features))[0]
        
        res_lag_1 = history_residuals[-1]
        res_lag_2 = history_residuals[-2]
        res_lag_3 = history_residuals[-3]
        
        lag_features = pd.DataFrame([[res_lag_1, res_lag_2, res_lag_3, feat_week]], columns=resid_features)
        
        pred_residual = model_resid.predict(lag_features)[0]
        pred_residual_upper = model_resid_upper.predict(lag_features)[0]
        pred_residual_lower = model_resid_lower.predict(lag_features)[0]
        
        final_pred = max(0, seasonal_base + pred_residual)
        upper_bound = max(0, seasonal_base + pred_residual_upper)
        lower_bound = max(0, seasonal_base + pred_residual_lower)

        history_residuals.append(pred_residual)
        
        future_forecasts.append({
            'date': current_date.strftime('%Y-%m-%d'),
            'Seasonal_Base': float(seasonal_base),
            'Final_Forecast': float(final_pred),
            'Lower_Bound': float(lower_bound),
            'Upper_Bound': float(upper_bound)
        })
        
    return future_forecasts, importance_list

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
        
        forecasts, importances = train_and_forecast(df)
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
                "baseline": [x['Seasonal_Base'] for x in forecasts],
                "lower": [x['Lower_Bound'] for x in forecasts],
                "upper": [x['Upper_Bound'] for x in forecasts]
            },
            "accuracy": accuracy_data,
            "importance": importances
        }
    else:
        print("  Skipping (No data found).")

with open('dashboard_data.json', 'w') as f:
    json.dump(full_dashboard_data, f)

print("\nSuccess: 'dashboard_data.json' updated with Residuals.")
