"""
Build the daily JSON files the volcanotech.co.uk dashboards read.

The format follows Tom's generate_web_data_reventador.py on the droplet, checked against
/srv/volcano/reventador/web/reventador_data_2025-12-06.json. The difference is that his
script reads one processing folder per day, while this merges every processed hour.
"""
import datetime
import json
import os
import re

import numpy as np
import pandas as pd

FLOW_TYPES = ['glob', 'histo', 'hybrid']
R2_BINS = [0, 0.3, 0.5, 0.8, 1.0]
R2_LABELS = ['Poor', 'Acceptable', 'Good', 'Excellent']


def load_emission_rates(path):
    """Read an EmissionRates file into a datetime / flux_kgs / flux_err table."""
    if not path.exists():
        return pd.DataFrame()

    header = None
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if header is None and 'datetime' in line:
            header = [name.strip() for name in line.replace('#', '').split(',')]
        elif header is not None and re.match(r'^\d{4}-\d{2}-\d{2}', line):
            rows.append([value.strip() for value in line.split(',')])

    if header is None or not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=header[:len(rows[0])])
    flux_col = None
    err_col = None
    for col in df.columns:
        if 'flux' in col and 'err' in col:
            err_col = col
        elif 'flux' in col and flux_col is None:
            flux_col = col

    out = pd.DataFrame()
    out['datetime'] = pd.to_datetime(df['datetime'])
    out['flux_kgs'] = pd.to_numeric(df[flux_col], errors='coerce') if flux_col else None
    out['flux_err'] = pd.to_numeric(df[err_col], errors='coerce') if err_col else 0.0
    return out.dropna(subset=['datetime', 'flux_kgs'])


def load_calibration(path):
    """Read the time, DOAS column density and R² columns of full_calibration.csv."""
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, skiprows=5)
    df = df[['timepoint', 'col density (doas)', 'r-squared']]
    df = df.rename(columns={'timepoint': 'time', 'col density (doas)': 'so2_col_dens', 'r-squared': 'r2'})
    df['time'] = pd.to_datetime(df['time'])
    return df


def summary_stats(df):
    flux = pd.to_numeric(df['flux_kgs'], errors='coerce').replace([np.inf, -np.inf], np.nan)
    if flux.isnull().all():
        return {'mean': None, 'median': None, 'min': None, 'max': None, 'q1': None, 'q3': None}
    return {'mean': flux.mean(), 'median': flux.median(), 'min': flux.min(), 'max': flux.max(),
            'q1': flux.quantile(0.25), 'q3': flux.quantile(0.75)}


def to_json_value(value):
    """Turn NaN, inf and NaT into None so the JSON is valid."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if value == 'NaT':
        return None
    return value


def build_day(processed_dirs, date):
    """
    Combine the pyplis output folders of one day into the dashboard data structure.
    :param processed_dirs:  list of Processed_<time> folders (one per hour)
    :param date:            datetime.date
    :returns:               dict ready for json.dump, or None if there is no emission data
    """
    file_date = date.strftime('%Y%m%d')
    emissions = []
    calibrations = []
    for folder in processed_dirs:
        for flow in FLOW_TYPES:
            df = load_emission_rates(folder / 'line_0' / f'{file_date}_EmissionRates_flow_{flow}.txt')
            if not df.empty:
                df['type'] = flow
                emissions.append(df)
        cal = load_calibration(folder / 'full_calibration.csv')
        if not cal.empty:
            calibrations.append(cal)

    if not emissions:
        return None

    data = pd.concat(emissions).sort_values('datetime').reset_index(drop=True)
    if calibrations:
        cal = pd.concat(calibrations).sort_values('time').reset_index(drop=True)
        data = pd.merge_asof(data, cal, left_on='datetime', right_on='time',
                             direction='nearest', tolerance=pd.Timedelta('30s'))
    else:
        data['time'] = None
        data['so2_col_dens'] = None
        data['r2'] = None

    # Tom's script turns a missing R² into the string 'nan'; the website expects that
    quality = pd.cut(pd.to_numeric(data['r2'], errors='coerce'), bins=R2_BINS, labels=R2_LABELS,
                     right=True, include_lowest=True)
    data['r2_quality'] = quality.astype(str)
    data['datetime'] = data['datetime'].astype(str)
    data['time'] = data['time'].astype(str)

    good = data[data['r2_quality'].isin(['Good', 'Excellent'])]
    stats = {'all_data': {}, 'good_data': {}}
    for flow in FLOW_TYPES:
        stats['all_data'][flow] = summary_stats(data[data['type'] == flow])
        stats['good_data'][flow] = summary_stats(good[good['type'] == flow])

    records = []
    for record in data.to_dict('records'):
        records.append({key: to_json_value(value) for key, value in record.items()})

    for group in stats.values():
        for flow, values in group.items():
            group[flow] = {key: to_json_value(value) for key, value in values.items()}

    return {
        'date': date.isoformat(),
        'last_updated': datetime.datetime.now().isoformat(),
        'time_series_data': records,
        'summary_stats': stats,
    }


def write_day(web_dir, prefix, day_data):
    """Write <prefix>_data_<date>.json and refresh the <prefix>_dates.json index."""
    web_dir.mkdir(parents=True, exist_ok=True)
    path = web_dir / f"{prefix}_data_{day_data['date']}.json"

    # Write to a temporary file first so the website never reads a half-written file
    tmp_path = path.with_suffix('.json.tmp')
    with open(tmp_path, 'w') as f:
        json.dump(day_data, f, indent=2)
    os.replace(tmp_path, path)

    dates = []
    for file in web_dir.glob(f'{prefix}_data_*.json'):
        match = re.search(r'_data_(\d{4}-\d{2}-\d{2})\.json$', file.name)
        if match:
            dates.append(match.group(1))
    dates.sort(reverse=True)
    with open(web_dir / f'{prefix}_dates.json', 'w') as f:
        json.dump(dates, f, indent=2)

    return path
