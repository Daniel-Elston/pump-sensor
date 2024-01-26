# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from utils.file_load import FileLoader
from utils.file_log import Logger


class BaseDataProcessing:
    def __init__(self, data_path, config, time_window, method='normalize'):
        """
        Initialize the BaseDataProcessing with a data path.

        Args:
            data_path (str): Path to the data.
        """
        self.logger = Logger(
            'BaseProcessLog', f'{Path(__file__).stem}.log').get_logger()
        self.data_path = data_path
        self.config = config
        self.raw_data = None
        self.df = None

        self.method = method
        self.time_window = config['time_window']
        self.scaler = StandardScaler() if method == 'standardize' else MinMaxScaler()

    def load_data(self):
        """
        Load the data.
        """
        self.df = FileLoader().load_file(self.data_path)
        self.logger.info(f'Initial dataset shape {self.df.shape}')
        self.logger.debug(f'Initial dataset sample:\n{self.df.sample(5)}')

    def select_data(self):
        """
        Select the sensor data.
        """
        self.df = self.df.iloc[:, 1:-1]
        self.df = self.df.drop(self.config['cols_to_drop'], axis=1)
        self.df = self.df[['timestamp'] + self.config['sensors_to_use']]

    # BASIC PROCESSING
    def handle_missing_values(self):
        """
        Handle missing values in the data.
        """
        self.df.ffill(inplace=True)

    def handle_nan_values(self):
        """
        Handle NaN values in the data.
        """
        self.df.ffill(inplace=True)

    # ADVANCED PROCESSING
    def scale_data(self):
        """
        Scale the data.
        """
        timestamp_col = self.df['timestamp']

        # Drop the timestamp column before scaling
        to_scale = self.df.drop('timestamp', axis=1)

        # Scale the data
        scaled_data = self.scaler.fit_transform(to_scale.values)

        # Create a new DataFrame with scaled data
        scaled_df = pd.DataFrame(scaled_data, columns=to_scale.columns)

        # Reattach the timestamp column to the scaled DataFrame
        self.df = pd.concat([timestamp_col, scaled_df], axis=1)
        self.logger.info(f'Dataset scaled using {self.method}')

    # TIMESERIES

    def resample_datetime(self):
        """
        Resample the data by datetime.
        """
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])

        self.df = self.df.set_index('timestamp')
        self.df = self.df.resample(self.time_window).mean()
        self.df = self.df.reset_index()
        self.logger.info(f'Dataset resampled by {self.time_window}')

    def convert_data_types(self):
        """
        Convert data types to appropriate formats.
        """
        # self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df['timestamp'] = self.df['timestamp'].astype(
            'int64') // 10**9  # to UNIX
        self.logger.info('Timestamp converted to UNIX')

    def sort_by_timestamp(self):
        """
        Sort the data by the timestamp column.
        """
        self.df.sort_values(by='timestamp', inplace=True)

    # PIPELINE
    def process(self):
        """
        Perform initial processing on the data.
        """
        self.load_data()
        self.select_data()
        self.handle_missing_values()
        self.resample_datetime()
        self.convert_data_types()
        # self.sort_by_timestamp()
        self.scale_data()
        print(self.df.shape)

        self.logger.info('Base Processing completed')
        self.logger.info(f"Final dataset shape: {self.df.shape}")
        self.logger.debug(f"Final dataset sample:\n{self.df.sample(5)}")

        return self.df
