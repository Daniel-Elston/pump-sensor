# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from pathlib import Path

import dotenv
import pandas as pd
import yaml
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

project_dir = Path(__file__).resolve().parents[2]
dotenv.load_dotenv(os.path.join(project_dir, '.env'))
config_path = os.path.join(project_dir, 'my_config.yaml')


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


class DataSelect:
    def __init__(self, df, config):
        """
        Initialize the DataSelect with a DataFrame.

        Args:
            df (DataFrame): The sensor data.
            config (dict): Configuration parameters.
        """
        self.df = df
        self.config = config

    def loc_data(self):
        """
        Select the sensor data.
        """
        self.df = self.df.iloc[:, 1:-1]

    def drop_data(self):
        """
        Drop the sensor data.
        """
        self.df = self.df.drop(self.config['cols_to_drop'], axis=1)

    def select_sensor(self):
        """
        Select which sensors to use
        """
        self.df = self.df[['timestamp'] + self.config['sensors_to_use']]

    def select_data(self):
        self.loc_data()
        self.drop_data()
        self.select_sensor()
        return self.df


class InitialProcessor:
    def __init__(self, df):
        """
        Initialize the InitialProcessor with a DataFrame.

        Args:
            df (DataFrame): The sensor data.
        """
        self.df = df

    def handle_missing_values(self):
        """
        Handle missing values in the data.
        """
        self.df.ffill(inplace=True)

    def convert_data_types(self):
        """
        Convert data types to appropriate formats.
        """
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])

    def sort_by_timestamp(self):
        """
        Sort the data by the timestamp column.
        """
        self.df.sort_values(by='timestamp', inplace=True)

    def process(self):
        """
        Perform initial processing on the data.
        """
        self.handle_missing_values()
        self.convert_data_types()
        # self.sort_by_timestamp()
        return self.df


class AdvancedProcessor:
    def __init__(self, df, method='standardize'):
        """
        Initialize the AdvancedProcessor with a DataFrame.

        Args:
            df (DataFrame): The sensor data.
            method (str): Method for scaling ('standardize' or 'normalize').
        """
        self.df = df
        self.method = method
        self.scaler = StandardScaler() if method == 'standardize' else MinMaxScaler()

    def scale_data(self):
        """
        Scale the sensor data using the specified method.
        """
        # Store the timestamp column in a separate variable
        timestamp_col = self.df['timestamp']

        # Drop the timestamp column before scaling
        to_scale = self.df.drop('timestamp', axis=1)

        # Scale the data
        scaled_data = self.scaler.fit_transform(to_scale.values)

        # Create a new DataFrame with scaled data
        scaled_df = pd.DataFrame(scaled_data, columns=to_scale.columns)

        # Reattach the timestamp column to the scaled DataFrame
        self.df = pd.concat([timestamp_col, scaled_df], axis=1)

        return self.df

    def process(self):
        """
        Perform advanced processing on the data.
        """
        return self.scale_data()
