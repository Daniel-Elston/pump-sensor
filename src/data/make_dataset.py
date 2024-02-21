# -*- coding: utf-8 -*-
from __future__ import annotations

import logging

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class SensorDataset(Dataset):
    def __init__(self, config, data):
        self.config = config
        self.data = data
        self.detection_alg = config['detection_alg']
        self.processed_items = 0
        self.log = logging.getLogger(self.__class__.__name__)

    def prepare_iso_data(self):
        self.log.info("Preparing data for Isolation Forest...")
        self.dataloader = DataLoader(
            self, batch_size=self.config['batch_size'], shuffle=False)
        sensor_data_list = []

        for batch in self.dataloader:
            sensor_data = batch[:, 1:]  # Exclude the timestamp column
            sensor_data_list.append(sensor_data)

        sensor_data_np = torch.cat(sensor_data_list).numpy()
        self.log.info("Converted Isolation Forest data to NumPy array.")
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
        self.log.info(
            f"Converted dataset of length {self.processed_items} to tensor")
        self.log.info(f"Dataset shape: {self.data.shape}")
        self.log.info(f"Dataset sample:\n{self.data.head()}")
