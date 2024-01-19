# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.data.base_processing import BaseDataProcessing
from utils.file_log import Logger


class SensorDataset(Dataset, BaseDataProcessing):
    def __init__(self, data_path, config, method='normalize', time_window='1min'):
        """
        Initialize the dataset object.
        Args:
            data_path (str): Path to the data.
            config (dict): Configuration parameters.
            method (str): Method to use for scaling the data.
        """
        self.logger = Logger(
            'MakeDatasetLog', f'{Path(__file__).stem}.log').get_logger()
        BaseDataProcessing.__init__(
            self, data_path, config, method=method, time_window=config['time_window'])
        self.data = self.process()  # Use parents' process method and store
        self.config = config
        self.time_window = time_window
        self.processed_items = 0

    def __getitem__(self, col):
        """
        Retrieve a single sample from the dataset.
        Args:
            col (int): Column id of the sensor data to retrieve.
        Returns:
            torch.Tensor: A tensor representing a sequence of sensor readings.
        """
        # Convert col to PyTorch tensor
        sensor_data = torch.tensor(
            self.data.iloc[col].values, dtype=torch.float)
        self.processed_items += 1
        return sensor_data

    def __len__(self):
        return len(self.data)

    def log_summary(self):
        self.logger.info(f'Converting dataset of len {
                         self.processed_items} to tensor')
        self.logger.info(f'Dataset shape: {self.data.shape}')
        self.logger.info(f'Dataset sample:\n{self.data.head()}')
