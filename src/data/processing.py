# -*- coding: utf-8 -*-
from __future__ import annotations

import logging

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


class Processor:
    def __init__(self, config):
        self.config = config
        self.sensor_to_use = config['sensors_to_use']
        self.method = config['scaler']
        self.time_window = config['time_window']

        self.logger = logging.getLogger(self.__class__.__name__)

    def select_data(self, df):
        """Select the sensor data."""
        df = df.iloc[:, 1:-1]
        df = df.drop(self.config['cols_to_drop'], axis=1)
        df = df[['timestamp'] + self.sensor_to_use]
        return df

    def handle_missing_values(self, df):
        df.ffill(inplace=True)
        return df

    def scale_data(self, df):
        timestamp_col = df['timestamp']
        to_scale = df.drop('timestamp', axis=1)
        scaler = StandardScaler() if self.method == 'standardize' else MinMaxScaler()
        scaled_data = scaler.fit_transform(to_scale.values)
        scaled_df = pd.DataFrame(scaled_data, columns=to_scale.columns)
        df = pd.concat([timestamp_col, scaled_df], axis=1)
        return df

    def resample_datetime(self, df):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        df = df.resample(self.time_window).mean()
        df = df.reset_index()
        return df

    def convert_data_types(self, df):
        df['timestamp'] = df['timestamp'].astype('int64') // 10**9
        return df

    def sort_by_timestamp(self, df):
        df.sort_values(by='timestamp', inplace=True)
        return df

    def run_processing(self, df):
        self.logger.debug(f"Shape: {df.shape}, type: {type(df)}")

        df = self.select_data(df)
        df = self.handle_missing_values(df)
        df = self.resample_datetime(df)
        df = self.convert_data_types(df)
        # self.sort_by_timestamp(df)
        self.logger.debug(f"Shape: {df.shape}, type: {type(df)}")

        df = self.scale_data(df)
        self.logger.debug(f"Shape: {df.shape}, type: {type(df)}")
        return df
