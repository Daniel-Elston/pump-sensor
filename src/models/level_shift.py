from __future__ import annotations

import pandas as pd
import ruptures as rpt
from adtk.data import validate_series
from adtk.detector import PersistAD


class LevelShiftDetector:
    def __init__(self, config, df):
        self.config = config
        self.df = df
        self.sensor_n = config['sensor_n']
        self.window = config['window']
        self.c = config['c']
        self.n_bkps = config['n_bkps']

    def detect_shifts(self):
        """
        Detect level shifts in a time series.

        Returns:
            pd.DataFrame: The input DataFrame with an added 'alarm' column indicating level shifts.
        """
        if self.config['shift_alg'] == 'ruptures':
            return self.ruptures_level_shift()
        elif self.config['shift_alg'] == 'adtk':
            return self.adtk_level_shift()
        else:
            raise ValueError('Invalid level shift algorithm')

    def ruptures_level_shift(self, model="rbf"):
        """
        Detect level shifts in a time series using the ruptures library.

        Args:
            model (str): The model used for detection.

        Returns:
            pd.DataFrame: The input DataFrame with an added 'alarm' column indicating level shifts.
        """
        series = self.df[f'sensor_{self.sensor_n}']
        algo = rpt.Binseg(model=model, min_size=75)
        algo.fit(series.values)
        result = algo.predict(n_bkps=self.n_bkps)

        alarms = pd.Series(False, index=series.index)
        for cp in result[:-1]:  # Skip the last one as it is just the end of the series
            alarms.iloc[cp] = True
        self.df['alarm'] = alarms
        return self.df[self.df.alarm]

    def adtk_level_shift(self, side='negative'):
        """
        Detect level shift using adtk library

        Args:
            side (str): The side for the PersistAD algorithm.

        Returns:
            pd.Series: The series of alarms.
        """
        s = validate_series(self.df[f'sensor_{self.sensor_n}'])

        persist_ad = PersistAD(c=self.c, side=side)
        persist_ad.window = self.window
        anomalies = persist_ad.fit_detect(s)

        anomalies = anomalies.fillna(False)
        return anomalies[anomalies]
