from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from torchinfo import summary


def dataset_stats(data):
    return data.mean(), data.std(), data.min(), data.max()


def str_contains(df, col, pattern):
    mask = df[col].str.contains(pattern, na=False)
    return df[mask]


def check_for_missing_dates(df):
    date_range = pd.date_range(
        start=df.date.min(), end=df.date.max(), freq='D')
    missing_dates = date_range.difference(df.date)
    return missing_dates


def pressure_to_kPa(df1, df2):
    df1[['avg_sea_level_pres_hpa']] = df1[['avg_sea_level_pres_hpa']] / 10
    df2[['pressure']] = df2[['pressure']] / 1000
    return df1, df2


def n_nans(df):
    for col in df.columns:
        df_nans = df[col].isna().sum()
        print(f'{col}: {df_nans}, {round(((df_nans/len(df))*100), 3)}')


def save_model_results(config, opt_name, params, run_results: dict):
    results_path = Path(config['results_path'])
    results = []

    if os.path.exists(results_path):
        with open(results_path, 'r') as file:
            results = json.load(file)

    results.append({
        'run_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'optimiser': opt_name,
        'lr': params.lr,
        'weight_decay': params.weight_decay,
        'momentum': params.momentum,
        'dampeing': params.dampening,
        'run_results': run_results,
    })

    with open(results_path, 'w') as file:
        json.dump(results, file, indent=4)


def save_model_summary(config, model):
    with open(Path(config['summary_path']), 'w', encoding='utf-8') as f:
        print(summary(model, (1, 1, config['window_size'])), file=f)
