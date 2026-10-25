import pandas as pd
import numpy as np
import json
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib.parse import quote
import re
import time
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.inspection import permutation_importance

# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------
# Each metric lists the primary UKHSA metric ID plus a 'fallback_pattern' regex.
# UKHSA periodically renames/replaces metrics (e.g. when surveillance methodology
# changes between seasons), which silently freezes the old series. At runtime the
# script lists the metrics the API actually offers for the topic and, if a metric
# matching the pattern has fresher data than the configured ID, switches to it.
METRICS = {
    "positivity": {
        "topic": "Influenza",
        "metric_id": "influenza_testing_positivityByWeek",
        "fallback_pattern": r"(?i)influenza.*test.*positivity",
        "agg": "mean",
        "name": "Flu: PCR Positivity Rate (%)"
    },
    "hospital": {
        "topic": "Influenza",
        "metric_id": "influenza_healthcare_hospitalAdmissionRateByWeek", 
        "fallback_pattern": r"(?i)influenza(?!.*(?:icu|hdu)).*admission.*rate",
        "agg": "mean",
        "name": "Flu: Hospital Admission Rate"
    },
    "icu": {
        "topic": "Influenza",
        "metric_id": "influenza_healthcare_ICUHDUadmissionRateByWeek",
        "fallback_pattern": r"(?i)influenza.*(?:icu|hdu).*admission.*rate",
        "agg": "mean",
        "name": "Flu: ICU/HDU Admission Rate"
    },
    "covid_positivity": {
        "topic": "COVID-19",
        "metric_id": "COVID-19_testing_positivity7DayRolling",
        "fallback_pattern": r"(?i)covid.*test.*positivity",
        "agg": "mean",
        "name": "COVID: PCR Positivity Rate (%)"
    },
    "covid_hospital": {
        "topic": "COVID-19",
        "metric_id": "COVID-19_healthcare_admissionByDay", 
        "fallback_pattern": r"(?i)covid.*healthcare.*admission(?!.*rate)",
        "agg": "sum",
        "name": "COVID: Weekly Hospital Admissions"
    }
}

# Headline/summary metrics are single snapshot numbers (not time series) and must
# never be selected as a fallback for a weekly series.
FALLBACK_EXCLUDE_PATTERN = re.compile(r"(?i)headline|change|percent|latest")

API_TEMPLATE = "https://api.ukhsa-dashboard.data.gov.uk/themes/infectious_disease/sub_themes/respiratory/topics/{topic}/geography_types/Nation/geographies/England/metrics/{metric_id}"
METRICS_LIST_TEMPLATE = "https://api.ukhsa-dashboard.data.gov.uk/themes/infectious_disease/sub_themes/respiratory/topics/{topic}/geography_types/Nation/geographies/England/metrics"

# key -> metric ID whose data was actually used for the national series, so the
# regional map can query the same (possibly renamed) metrics.
RESOLVED_METRIC_IDS = {}

# Approximate centroid coordinates for the nine UKHSA Regions. Regions returned by
# the API that are not listed here fall back to a central England coordinate.
REGION_COORDS = {
    "London": (51.5074, -0.1278),
    "South East": (51.2564, -0.7252),
    "South West": (50.8000, -3.8000),
    "East of England": (52.2400, 0.4000),
    "West Midlands": (52.4800, -1.9000),
    "East Midlands": (52.9500, -0.9500),
    "North West": (53.4800, -2.2400),
    "Yorkshire and The Humber": (53.8000, -1.5500),
    "North East": (54.9700, -1.6100),
}

# Region map keys -> the national METRICS entry whose (possibly re-resolved)
# metric ID should be queried at UKHSA Region level.
REGION_MAP_METRICS = {
    "pos_value": "positivity",
    "adm_value": "hospital",
    "icu_value": "icu",
}

REGION_BASE_URL = (
    "https://api.ukhsa-dashboard.data.gov.uk/themes/infectious_disease/sub_themes/"
    "respiratory/topics/Influenza/geography_types/UKHSA%20Region/geographies"
)


def fetch_latest_metric(session, url, headers, region_name, metric_id):
    current_url = url
    params = {'page_size': 365, 'format': 'json'}
    latest_result = None

    while current_url:
        try:
            response = session.get(current_url, headers=headers, params=params, timeout=10)
            params = None
            if response.status_code != 200:
                print(f"    Warning: HTTP {response.status_code} fetching '{metric_id}' for '{region_name}'.")
                # Return whatever was already found on earlier pages rather than
                # discarding valid data because a later page failed.
                return latest_result

            payload = response.json()
            for result in payload.get('results') or []:
                result_date = result.get('date')
                if result_date and (latest_result is None or result_date > latest_result.get('date', '')):
                    latest_result = result
            current_url = payload.get('next')
            time.sleep(0.1)
        except Exception as e:
            print(f"    Warning: Error fetching '{metric_id}' for '{region_name}': {e}")
            # As above: a transient error on one page shouldn't wipe out results
            # already collected from previous pages.
            return latest_result

    return latest_result


def fetch_region_influenza_map_data(days=7):
    """Fetch the latest Influenza positivity, hospital and ICU/HDU rates per UKHSA Region."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json'
    }

    retry = Retry(total=2, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session = requests.Session()
    session.mount('https://', HTTPAdapter(max_retries=retry))

    try:
        resp = session.get(REGION_BASE_URL, headers=headers, params={'format': 'json'}, timeout=15)
        if resp.status_code != 200:
            print(f"  [HTTP {resp.status_code}] Could not fetch UKHSA Region list.")
            return None
        payload = resp.json()
    except Exception as e:
        print(f"  [Error] Could not fetch UKHSA Region list: {e}")
        return None

    regions = payload if isinstance(payload, list) else payload.get('results', [])
    region_names = [r['name'] for r in regions if isinstance(r, dict) and 'name' in r]
    if not region_names:
        print("  [Error] UKHSA Region list was empty.")
        return None

    end_date = pd.Timestamp.utcnow().normalize().tz_localize(None)
    start_date = end_date - pd.Timedelta(days=days)
    date_range_str = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"

    points = []
    for region_name in region_names:
        lat, lon = REGION_COORDS.get(region_name, (54.0, -2.0))
        record = {'region': region_name, 'lat': lat, 'lon': lon, 'date': date_range_str}
        has_value = False

        for key, metric_key in REGION_MAP_METRICS.items():
            record[key] = None
            configured_id = METRICS[metric_key]['metric_id']
            resolved_id = RESOLVED_METRIC_IDS.get(metric_key, configured_id)
            # Try the metric actually used nationally first, then the configured one.
            metric_ids = [resolved_id] if resolved_id == configured_id else [resolved_id, configured_id]
            latest_result = None
            for metric_id in metric_ids:
                url = f"{REGION_BASE_URL}/{quote(region_name)}/metrics/{metric_id}"
                latest_result = fetch_latest_metric(session, url, headers, region_name, metric_id)
                if latest_result:
                    break
            if not latest_result:
                continue
            value = pd.to_numeric(latest_result.get('metric_value'), errors='coerce')
            if pd.isna(value):
                continue
            record[key] = float(value)
            record[f"{key}_date"] = latest_result.get('date')
            has_value = True
            time.sleep(0.1)

        if has_value:
            points.append(record)

    if not points:
        return None

    return {'points': points, 'date_range': date_range_str}


def _build_session():
    retry = Retry(total=5, backoff_factor=2, status_forcelist=[403, 429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


_API_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'application/json'
}

_available_metrics_cache = {}


def list_available_metrics(session, topic):
    """List the metric IDs the UKHSA API currently offers for a topic (England, Nation level)."""
    if topic in _available_metrics_cache:
        return _available_metrics_cache[topic]

    names = []
    current_url = METRICS_LIST_TEMPLATE.format(topic=topic)
    params = {'format': 'json'}
    while current_url:
        try:
            response = session.get(current_url, headers=_API_HEADERS, params=params, timeout=15)
            params = None
            if response.status_code != 200:
                print(f"    Warning: HTTP {response.status_code} listing metrics for '{topic}'.")
                break
            payload = response.json()
            results = payload if isinstance(payload, list) else payload.get('results', [])
            for item in results or []:
                if isinstance(item, dict) and item.get('name'):
                    names.append(item['name'])
            current_url = payload.get('next') if isinstance(payload, dict) else None
            time.sleep(0.2)
        except Exception as e:
            print(f"    Warning: Could not list metrics for '{topic}': {e}")
            break

    _available_metrics_cache[topic] = names
    return names


def fetch_metric_series(session, topic, metric_id, agg):
    """Fetch one metric as a weekly-resampled DataFrame, or None if no data."""
    all_data = []
    url = API_TEMPLATE.format(topic=topic, metric_id=metric_id)
    current_url = f"{url}?page_size=365&format=json"

    while current_url:
        try:
            response = session.get(current_url, headers=_API_HEADERS, timeout=15)
            if response.status_code != 200:
                print(f"    Warning: HTTP {response.status_code} for {metric_id}: {response.text[:250]}")
                return None
            data = response.json()
            all_data.extend(data['results'])
            current_url = data['next']
            time.sleep(0.5)
        except Exception as e:
            print(f"    Error fetching data: {e}")
            break

    if not all_data:
        return None

    df = pd.DataFrame(all_data)
    df = df[['date', 'metric_value']].copy()
    df['date'] = pd.to_datetime(df['date'])
    df['metric_value'] = pd.to_numeric(df['metric_value'], errors='coerce')
    
    if agg == 'sum':
        df = df.groupby('date')['metric_value'].sum(min_count=1).reset_index()
    else:
        df = df.groupby('date')['metric_value'].mean().reset_index()

    df.sort_values('date', inplace=True)
    df.set_index('date', inplace=True)
    
    if agg == 'sum':
        df = df.resample('W-SUN')['metric_value'].sum(min_count=1).to_frame()
    else:
        df = df.resample('W-SUN')['metric_value'].mean().to_frame()
    
    df = df[df['metric_value'].notna()].copy()

    if df.empty:
        return None

    return df


def fetch_data(key, config):
    session = _build_session()
    metric_id = config['metric_id']
    print(f"  Fetching {config['topic']} data: {metric_id}...")
    primary_df = fetch_metric_series(session, config['topic'], metric_id, config['agg'])
    if primary_df is not None:
        print(f"    Latest data point for '{metric_id}': {primary_df.index[-1].date()}")

    # Discover alternative metric IDs in case UKHSA renamed/replaced the series.
    pattern = re.compile(config['fallback_pattern'])
    available = list_available_metrics(session, config['topic'])
    candidates = [
        m for m in available
        if m != metric_id and pattern.search(m) and not FALLBACK_EXCLUDE_PATTERN.search(m)
    ]

    best_df, best_id = primary_df, metric_id
    for candidate in candidates:
        print(f"  Checking alternative metric: {candidate}...")
        candidate_df = fetch_metric_series(session, config['topic'], candidate, config['agg'])
        if candidate_df is None:
            continue
        print(f"    Latest data point for '{candidate}': {candidate_df.index[-1].date()}")
        if best_df is None or candidate_df.index[-1] > best_df.index[-1]:
            best_df, best_id = candidate_df, candidate

    if best_df is None:
        return None

    if best_id != metric_id:
        print(f"    Using fresher metric '{best_id}' instead of configured '{metric_id}'.")
        # Keep enough history for model training by splicing the legacy series
        # in front of the replacement series.
        if primary_df is not None:
            legacy = primary_df[primary_df.index < best_df.index[0]]
            if not legacy.empty:
                print(f"    Splicing {len(legacy)} legacy weeks from '{metric_id}' before {best_df.index[0].date()}.")
                best_df = pd.concat([legacy, best_df]).sort_index()

    RESOLVED_METRIC_IDS[key] = best_id
    return best_df

def evaluate_accuracy(df, weeks_back=52):
    if len(df) < (weeks_back + 20):
        weeks_back = max(1, len(df) - 20)

    dates = []
    actuals = []
    predictions = []
    residuals = []

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
        residuals.append(float(actual_value - final_pred))

    if not actuals: return None
    
    mae = mean_absolute_error(actuals, predictions)
    mape = np.mean(np.abs((np.array(actuals) - np.array(predictions)) / np.maximum(np.array(actuals), 1))) * 100

    return {
        "dates": dates,
        "actuals": actuals,
        "predictions": predictions,
        "residuals": residuals,
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
# Maximum gap (in days) allowed between the latest UKHSA data point for a metric
# and the freshest data point retrieved across all metrics in this run before the
# metric is treated as "stale" and a brief explanation is shown under its chart.
# 21 days (3 weeks) comfortably covers UKHSA's off-season fortnightly reporting
# cadence while still catching genuine outages/discontinuations like multi-month gaps.
STALE_DATA_THRESHOLD_DAYS = 21

full_dashboard_data = {}
last_dates = {}  # key -> most recent pandas Timestamp successfully retrieved
print("Starting Multi-Virus Forecast Job...")

for key, config in METRICS.items():
    print(f"\nProcessing: {config['name']}")
    df = fetch_data(key, config)
    
    if df is not None and not df.empty:
        if len(df) < 20:
             print("  Skipping (Not enough data points).")
             continue
        
        forecasts, importances = train_and_forecast(df)
        accuracy_data = evaluate_accuracy(df, weeks_back=52)

        last_dates[key] = df.index[-1]
        print(f"  Using metric '{RESOLVED_METRIC_IDS.get(key, config['metric_id'])}' (latest: {df.index[-1].date()}).")
        full_dashboard_data[key] = {
            "meta": {
                "name": config['name'],
                "topic": config['topic'],
                "metric_id": RESOLVED_METRIC_IDS.get(key, config['metric_id'])
            },
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
        full_dashboard_data[key] = {
            "meta": {"name": config['name'], "topic": config['topic']},
            "data_notice": (
                f"Live data unavailable: the UKHSA API did not return any data for "
                f"'{config['metric_id']}'. This may be a temporary API issue, or UKHSA "
                f"may have retired/renamed this metric."
            )
        }

# Flag any metric whose latest data point is significantly older than the freshest
# data retrieved in this run, which indicates UKHSA has stopped publishing (or is
# delayed in publishing) new figures for that specific series rather than a bug in
# this script. Surface a brief, honest explanation under the affected chart instead
# of silently showing out-of-date numbers.
if last_dates:
    newest_date = max(last_dates.values())
    for key, metric_last_date in last_dates.items():
        gap_days = (newest_date - metric_last_date).days
        if gap_days > STALE_DATA_THRESHOLD_DAYS:
            config = METRICS[key]
            full_dashboard_data[key]["data_notice"] = (
                f"Live data unavailable: UKHSA has not published a newer figure for "
                f"'{config['name']}' since {metric_last_date.strftime('%d %B %Y')} "
                f"({gap_days} days behind the most recent UKHSA update used elsewhere on "
                f"this dashboard). The chart shows the most recent data UKHSA has made "
                f"available via its API."
            )
            print(f"  Notice: '{config['name']}' is stale ({gap_days} days behind latest).")

# Optional UKHSA Region influenza maps. Isolated so that API timeouts or missing map
# data can never abort the forecasting job above.
try:
    print("\nProcessing: Influenza Metrics by UKHSA Region (maps)")
    region_map = fetch_region_influenza_map_data()
    if region_map:
        full_dashboard_data['region_flu_map'] = {
            "meta": {"name": "Influenza Metrics by UKHSA Region", "topic": "Influenza"},
            "date_range": region_map['date_range'],
            "points": region_map['points']
        }
        print(f"  Retrieved {len(region_map['points'])} regions.")
    else:
        print("  Skipping maps (No UKHSA Region data found).")
except Exception as e:
    print(f"  Skipping maps (Error building UKHSA Region map data: {e})")

with open('dashboard_data.json', 'w') as f:
    json.dump(full_dashboard_data, f)

print("\nSuccess: 'dashboard_data.json' updated with Residuals.")
