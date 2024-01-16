# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt

from src.data.make_dataset import SensorDataset
from src.models.iso_forest import IsolationForestAD
from src.models.level_shift import LevelShiftDetector
from src.visualization.visualize import apply_level_shifts
from src.visualization.visualize import create_df
from src.visualization.visualize import get_visuals
from utils.file_log import Logger
from utils.file_save import FileSaver
from utils.setup_env import setup_project_env
warnings.filterwarnings(action='ignore', category=FutureWarning)


logger = Logger('PipelineLogger', f'{Path(__file__).stem}.log').get_logger()


def main():
    logger.info("Pipeline started")

    # Set up environment
    project_dir, config = setup_project_env()
    logger.info("Environment and configuration setup completed")

    data_path = os.path.join(project_dir, 'data/sdo/sensor.parq')
    results_path = os.path.join(project_dir, 'results/iso2.json')

    # Pipeline
    try:
        dataset = SensorDataset(
            data_path, config, time_window=config['time_window'])
        logger.info(f'Dataset loaded from {data_path}')

        anomaly_model = IsolationForestAD(dataset, config)
        prepared_data = anomaly_model.prepare_data()

        anomalies, scores = anomaly_model.detect_anomalies(prepared_data)
        logger.info('Anomaly detection completed')

        df = create_df(config['sensor_n'], dataset,
                       prepared_data, anomalies, scores)

        shift_detector = LevelShiftDetector(config, df)
        alarms = shift_detector.ruptures_level_shift()
        # alarms = shift_detector.adtk_level_shift()
        logger.info('Level shift detection completed')

        # Visuals
        get_visuals(config['sensor_n'], df)
        apply_level_shifts(alarms, shift_type='ruptures')
        plt.show()
        logger.info('Visuals created')

        # Save results
        FileSaver().save_file(anomalies, results_path)
        logger.info(f'Anomaly detection results saved to {results_path}')

    except Exception as e:
        logger.error(f'Error in pipeline: {e}', exc_info=True)

    logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()
