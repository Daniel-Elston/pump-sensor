# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from src.data.base_processing import BaseDataProcessing
from utils.file_log import Logger


class SensorDataset(Dataset, BaseDataProcessing):
    def __init__(self, data_path, config, method='normalize'):
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
        self.dataloader = DataLoader(
            self, batch_size=config['batch_size'], shuffle=False)
        self.config = config
        self.time_window = config['time_window']
        self.processed_items = 0
        self.detection_algorithm = config['detection_alg']
        self.data = self.process()  # Use parents' process method and store

        # if self.detection_alg == 'iso':
        #     self.data = self.prepare_iso_data()
        # elif self.detection_alg == 'lstm':
        #     self.data = self.prepare_lstm_data()

    def prepare_iso_data(self):
        """
        Prepare the data for anomaly detection, usable by sklearn.
        """
        sensor_data_list = []

        for batch in self.dataloader:
            sensor_data = batch[:, 1:]  # Exclude the timestamp column
            sensor_data_list.append(sensor_data)

        # Convert list of tensors to a single numpy array
        sensor_data_np = torch.cat(sensor_data_list).numpy()
        return sensor_data_np

    def prepare_lstm_data(self):
        pass

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
        self.processed_items += 1  # Counter for logging
        return sensor_data

    def __len__(self):
        return len(self.data)

    def log_summary(self):
        self.logger.info(f'Converting dataset of len {
                         self.processed_items} to tensor')
        self.logger.info(f'Dataset shape: {self.data.shape}')
        self.logger.info(f'Dataset sample:\n{self.data.head()}')
