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
        self.config = config
        self.logger = Logger('MakeDatasetLog', f'{
                             Path(__file__).stem}.log').get_logger()
        self.logger.info("Initializing SensorDataset...")
        BaseDataProcessing.__init__(
            self, data_path, config, method=method, time_window=config['time_window'])
        self.detection_alg = config['detection_alg']
        self.processed_items = 0
        self.data = self.process()
        self.logger.info(f"Dataset loaded as DataFrame with shape: {
                         self.data.shape}")

    def prepare_iso_data(self):
        self.logger.info("Preparing data for Isolation Forest...")
        self.dataloader = DataLoader(
            self, batch_size=self.config['batch_size'], shuffle=False)
        sensor_data_list = []

        for batch in self.dataloader:
            sensor_data = batch[:, 1:]  # Exclude the timestamp column
            sensor_data_list.append(sensor_data)

        sensor_data_np = torch.cat(sensor_data_list).numpy()
        self.logger.info("Converted Isolation Forest data to NumPy array.")
        return sensor_data_np

    def __getitem__(self, idx):
        if self.detection_alg == 'iso':
            sensor_data = torch.tensor(
                self.data.iloc[idx].values, dtype=torch.float)
        else:
            raise ValueError('Invalid detection algorithm.')

        self.processed_items += 1
        return sensor_data

    def __len__(self):
        if self.detection_alg == 'iso':
            return len(self.data)
        else:
            raise ValueError('Invalid detection algorithm.')

    def log_summary(self):
        self.logger.info(f"Converted dataset of length {
                         self.processed_items} to tensor")
        self.logger.info(f"Dataset shape: {self.data.shape}")
        self.logger.info(f"Dataset sample:\n{self.data.head()}")
