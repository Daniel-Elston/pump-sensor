# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from pathlib import Path

import dotenv
import torch
import yaml
from base_processing import BaseDataProcessing
from torch.utils.data import Dataset

project_dir = Path(__file__).resolve().parents[2]
dotenv.load_dotenv(os.path.join(project_dir, '.env'))


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


class SensorDataset(Dataset, BaseDataProcessing):
    def __init__(self, data_path, config, method='normalize'):
        """
        Initialize the dataset object.
        Args:
            data_path (str): Path to the data.
            config (dict): Configuration parameters.
            method (str): Method to use for scaling the data.
        """
        BaseDataProcessing.__init__(self, data_path, config, method=method)
        self.data = self.process()  # Assuming this returns a DataFrame
        self.config = config

    def __getitem__(self, col):
        """
        Retrieve a single sample from the dataset.
        Args:
            col (int): Column id of the sesnor data to retrieve.
        Returns:
            torch.Tensor: A tensor representing a sequence of sensor readings.
        """
        # Convert row to PyTorch tensor
        numeric_data = self.data.select_dtypes(include=['number'])
        return torch.tensor(numeric_data.iloc[:, col].values, dtype=torch.float)

    def __len__(self):
        return len(self.data)


def main():
    data_path = os.path.join(project_dir, 'data/sdo/sensor.parq')
    config_path = os.path.join(project_dir, 'my_config.yaml')
    config = load_config(config_path)

    dataset = SensorDataset(data_path, config)

    print(dataset[0])


if __name__ == '__main__':
    main()
