from __future__ import annotations

import pandas as pd
import ruptures as rpt
from adtk.data import validate_series
from adtk.detector import PersistAD


def ruptures_level_shift(df, series, n_bkps=6*2, model="rbf"):
    """
    Detect level shifts in a time series using the ruptures library.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        series (pd.Series): The series in which to detect level shifts.
        n_bkps (int): Number of breakpoints to find.
        model (str): The model used for detection.

    Returns:
        pd.DataFrame: The input DataFrame with an added 'alarm' column indicating level shifts.
    """
    algo = rpt.Binseg(model=model, min_size=75)
    algo.fit(series.values)
    result = algo.predict(n_bkps=n_bkps)

    alarms = pd.Series(False, index=series.index)
    for cp in result[:-1]:  # Skip the last one as it is just the end of the series
        alarms.iloc[cp] = True
    df['alarm'] = alarms

    return df[df.alarm]


def adtk_level_shift(df, sensor_n, window=24, c=3.0, side='negative'):
    """
    Detect level shift using adtk library

    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        sensor_n (int): The sensor number to analyze.
        window (int): The window size for the PersistAD algorithm.
        c (float): The threshold for the PersistAD algorithm.
        side (str): The side for the PersistAD algorithm.

    Returns:
        pd.Series: The series of alarms.
    """
    s = validate_series(df[f'sensor_{sensor_n}'])

    persist_ad = PersistAD(c=c, side=side)
    persist_ad.window = window
    anomalies = persist_ad.fit_detect(s)

    anomalies = anomalies.fillna(False)
    return anomalies[anomalies]
