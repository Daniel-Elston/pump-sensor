# -*- coding: utf-8 -*-
from __future__ import annotations

import pandas as pd
import pyarrow.parquet as pq
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


class BaseDataProcessing:
    def __init__(self, data_path, config, method='normalize', time_window='1min'):
        """
        Initialize the BaseDataProcessing with a data path.

        Args:
            data_path (str): Path to the data.
        """
        self.data_path = data_path
        self.config = config
        self.raw_data = None
        self.df = None

        self.method = method
        self.scaler = StandardScaler() if method == 'standardize' else MinMaxScaler()
        self.time_window = time_window

    def load_data(self):
        """
        Load the data.
        """
        self.raw_data = pq.read_table(self.data_path).to_pandas()
        self.df = pq.read_table(self.data_path).to_pandas()

    def select_data(self):
        """
        Select the sensor data.
        """
        self.df = self.df.iloc[:, 1:-1]
        self.df = self.df.drop(self.config['cols_to_drop'], axis=1)
        self.df = self.df[['timestamp'] + self.config['sensors_to_use']]

    def handle_missing_values(self):
        """
        Handle missing values in the data.
        """
        self.df.ffill(inplace=True)

    def resample_datetime(self):
        """
        Resample the data by datetime.
        """
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])

        self.df = self.df.set_index('timestamp')
        self.df = self.df.resample(self.time_window).mean()
        self.df = self.df.reset_index()

    def convert_data_types(self):
        """
        Convert data types to appropriate formats.
        """
        # self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df['timestamp'] = self.df['timestamp'].astype('int64') // 10**9

    def sort_by_timestamp(self):
        """
        Sort the data by the timestamp column.
        """
        self.df.sort_values(by='timestamp', inplace=True)

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
        return self.df


# def main():
#     data_path = os.path.join(project_dir, 'data/sdo/sensor.parq')
#     config_path = os.path.join(project_dir, 'my_config.yaml')
#     config = load_config(config_path)

#     processor = BaseDataProcessing(data_path, config, config['time_window'])
#     processor.process()

#     print(processor.df.head())


# if __name__ == '__main__':
#     main()
