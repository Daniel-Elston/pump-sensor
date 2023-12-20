# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys
from pathlib import Path

import dotenv
import torch
from torch.utils.data import Dataset

from my_utils import load_config
from src.data.base_processing import BaseDataProcessing
# import pandas as pd
# import yaml
# from my_utils import save_to_parquet

project_dir = Path(__file__).resolve().parents[2]
dotenv.load_dotenv(os.path.join(project_dir, '.env'))
sys.path.append(str(project_dir))


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
        self.data = self.process()  # Assuming this returns a DataFrame
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


def main():
    data_path = os.path.join(project_dir, 'data/sdo/sensor.parq')
    config_path = os.path.join(project_dir, 'my_config.yaml')
    config = load_config(config_path)

    dataset = SensorDataset(data_path, config, config['time_window'])
    print(dataset[0].head())

    # # Save to Parquet
    # output_file_path = os.path.join(
    #     project_dir, 'data/processed/processed_sensor.parq')
    # save_to_parquet(df_final, output_file_path)
    # print(f"Data saved: {output_file_path}")


if __name__ == '__main__':
    main()
