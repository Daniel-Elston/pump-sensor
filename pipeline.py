# -*- coding: utf-8 -*-
from __future__ import annotations

import os

import matplotlib.pyplot as plt

from my_utils import load_config
from my_utils import save_to_json
from my_utils import setup_environment
from src.data.make_dataset import SensorDataset
from src.models.iso_forest import IsolationForestAD
from src.models.level_shift import ruptures_level_shift
from src.visualization.visualize import apply_level_shifts
from src.visualization.visualize import create_df
from src.visualization.visualize import get_visuals
# from src.models.level_shift import adtk_level_shift

sensor_n = 2


def main():

    # Set up environment
    project_dir = setup_environment()

    data_path = os.path.join(project_dir, 'data/sdo/sensor.parq')
    results_path = os.path.join(project_dir, 'results/iso1.json')
    config_path = os.path.join(project_dir, 'my_config.yaml')

    config = load_config(config_path)

    # Pipeline
    dataset = SensorDataset(
        data_path, config, time_window=config['time_window'])

    anomaly_model = IsolationForestAD(
        dataset, config, contamination=config['contamination'])
    prepared_data = anomaly_model.prepare_data()

    anomalies, scores = anomaly_model.detect_anomalies(prepared_data)

    df = create_df(sensor_n, dataset, prepared_data, anomalies, scores)

    alarms = ruptures_level_shift(df, df[f'sensor_{sensor_n}'])
    # alarms = adtk_level_shift(df, sensor_n)

    # Visuals
    get_visuals(sensor_n, df)
    apply_level_shifts(alarms, shift_type='ruptures')
    plt.show()

    # Save results
    save_to_json(anomalies, results_path)


if __name__ == "__main__":
    main()
