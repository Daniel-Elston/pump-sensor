# -*- coding: utf-8 -*-
from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.base_processing import BaseDataProcessing


class SensorDataset(Dataset, BaseDataProcessing):
    def __init__(self, data_path, config, method='normalize', time_window='1min'):
        """
        Initialize the dataset object.
        Args:
            data_path (str): Path to the data.
            config (dict): Configuration parameters.
            method (str): Method to use for scaling the data.
        """
        BaseDataProcessing.__init__(
            self, data_path, config, method=method, time_window=config['time_window'])
        self.data = self.process()  # Use parents' process method and store
        self.config = config
        self.time_window = time_window

    def __getitem__(self, col):
        """
        Retrieve a single sample from the dataset.
        Args:
            col (int): Column id of the sesnor data to retrieve.
        Returns:
            torch.Tensor: A tensor representing a sequence of sensor readings.
        """
        # Convert col to PyTorch tensor
        sensor_data = torch.tensor(
            self.data.iloc[col].values, dtype=torch.float)
        return sensor_data

    def __len__(self):
        return len(self.data)


def create_df(config, dataset, prepared_data, anomalies, scores):
    """
    Create a DataFrame from the sensor data, anomalies, and scores.

    Args:
        sensor_n (int): The sensor number to process.
        dataset (SensorDataset): The dataset containing the sensor data.
        prepared_data (np.array): The prepared sensor data.
        anomalies (np.array): Anomaly flags for the data.
        scores (np.array): Anomaly scores for the data.

    Returns:
        pd.DataFrame: A DataFrame with the processed data.
    """
    df = pd.DataFrame(
        {
            'timestamp': pd.to_datetime((dataset.data.timestamp.values), unit='s'),
            'unix': dataset.data.timestamp.values,
            f'sensor_{config['sensor_n']}': prepared_data[:, config['sensor_n']-1],
            'anomaly': anomalies[f'sensor_{config['sensor_n']}'],
            'score': scores[f'sensor_{config['sensor_n']}']
        }
    )
    df.set_index('timestamp', inplace=True)
    return df


# def main():
#     data_path = os.path.join(project_dir, 'data/sdo/sensor.parq')
#     config_path = os.path.join(project_dir, 'my_config.yaml')
#     config = load_config(config_path)

#     dataset = SensorDataset(data_path, config, config['time_window'])
#     print(dataset[0].head())

#     # # Save to Parquet
#     # output_file_path = os.path.join(
#     #     project_dir, 'data/processed/processed_sensor.parq')
#     # save_to_parquet(df_final, output_file_path)
#     # print(f"Data saved: {output_file_path}")


# if __name__ == '__main__':
#     main()
