# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from src.data.base_processing import BaseDataProcessing
from utils.file_log import Logger


class SensorDataset(Dataset, BaseDataProcessing):
    def __init__(self, data_path, config, method='normalize'):
        self.logger = Logger('MakeDatasetLog', f'{
                             Path(__file__).stem}.log').get_logger()
        self.logger.info("Initializing SensorDataset...")
        BaseDataProcessing.__init__(
            self, data_path, config, method=method, time_window=config['time_window'])
        self.config = config
        self.detection_alg = config['detection_alg']
        self.processed_items = 0
        self.data = self.process()
        self.segments = self.prepare_lstm_data()
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

    def prepare_lstm_data(self):
        self.logger.info("Preparing data for LSTM...")
        seq_length = self.config['seq_length']
        # change to seq_length - 1 = max overlap
        # overlap = self.config['overlap']
        data_array = self.data.values
        segments = []

        for start_pos in range(0, len(data_array) - seq_length + 1):
            end_pos = start_pos + seq_length
            segment = data_array[start_pos:end_pos, self.config['sensor_n']]
            # Reshape to [seq_length, 1]
            segments.append(segment.reshape(-1, 1))

        self.logger.info("Converted LSTM data to segmented NumPy array.")
        return np.array(segments)

    def __getitem__(self, idx):
        if self.detection_alg == 'iso':
            sensor_data = torch.tensor(
                self.data.iloc[idx].values, dtype=torch.float)
            return sensor_data
        elif self.detection_alg == 'lstm':
            segment = self.segments[idx]
            sequence = torch.tensor(segment, dtype=torch.float)
            return sequence
        else:
            raise ValueError('Invalid detection algorithm.')

        self.processed_items += 1
        # return sensor_data if self.detection_alg == 'iso' else sequence

    def __len__(self):
        if self.detection_alg == 'iso':
            return len(self.data)
        elif self.detection_alg == 'lstm':
            return len(self.segments)
        else:
            raise ValueError('Invalid detection algorithm.')

    def log_summary(self):
        self.logger.info(f"Converted dataset of length {
                         self.processed_items} to tensor")
        self.logger.info(f"Dataset shape: {self.data.shape}")
        self.logger.info(f"Dataset sample:\n{self.data.head()}")
