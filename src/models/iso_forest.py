# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import dotenv
import numpy as np
import yaml
from sklearn.ensemble import IsolationForest
from torch.utils.data import DataLoader

from src.data.make_dataset import SensorDataset

project_dir = Path(__file__).resolve().parents[2]
dotenv.load_dotenv(os.path.join(project_dir, '.env'))
sys.path.append(str(project_dir))


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


class IsoForestAD:
    def __init__(self, dataset, config, batch_size=1):
        self.dataset = dataset
        self.config = config
        self.dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False)

    def prepare_data(self):
        """
        Prepare the data for anomaly detection.
        """
        data_list = [self.dataset[i][1].numpy() for i in range(10)]
        return np.array(data_list)  # .reshape(-1, 1)

    def detect_anomalies(self, data):
        """
        Detect anomalies in the data.
        Args:
            data (np.ndarray): Numpy array of sensor data.
        Returns:
            np.ndarray: Array of anomaly labels.
        """
        clf = IsolationForest(random_state=42)
        clf.fit(data)
        return clf.predict(data)


def main():
    data_path = os.path.join(project_dir, 'data/sdo/sensor.parq')
    results_path = os.path.join(project_dir, 'results/iso1.json')

    config_path = os.path.join(project_dir, 'my_config.yaml')
    config = load_config(config_path)

    dataset = SensorDataset(data_path, config)

    anomaly_model = IsoForestAD(dataset, config)
    prepared_data = anomaly_model.prepare_data()

    anomalies = anomaly_model.detect_anomalies(prepared_data)
    # save results as JSON
    with open(results_path, 'w') as file:
        json.dump(anomalies.tolist(), file)


if __name__ == "__main__":
    main()
