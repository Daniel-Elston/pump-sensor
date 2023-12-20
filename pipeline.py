# -*- coding: utf-8 -*-
from __future__ import annotations

import os

from my_utils import load_config
from my_utils import save_to_json
from my_utils import setup_environment
from src.data.make_dataset import SensorDataset
from src.models.iso_forest import IsolationForestAD


def main():
    project_dir = setup_environment()

    data_path = os.path.join(project_dir, 'data/sdo/sensor.parq')
    results_path = os.path.join(project_dir, 'results/iso1.json')
    config_path = os.path.join(project_dir, 'my_config.yaml')

    config = load_config(config_path)

    dataset = SensorDataset(
        data_path, config, time_window=config['time_window'])

    anomaly_model = IsolationForestAD(
        dataset, config, contamination=config['contamination'])
    prepared_data = anomaly_model.prepare_data()

    anomalies = anomaly_model.detect_anomalies(prepared_data)

    save_to_json(anomalies, results_path)


if __name__ == "__main__":
    main()
