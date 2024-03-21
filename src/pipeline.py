# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.data.make_dataset import SensorDataset
from src.data.processing import Processor
from src.models.iso_forest import IsolationForestAD
from src.models.level_shift import LevelShiftDetector
from src.visualization.visualize import Visualiser
from utils.file_load import FileLoader
from utils.file_save import FileSaver
from utils.setup_env import setup_project_env
warnings.filterwarnings(action='ignore', category=FutureWarning)


class DataPipeline:
    def __init__(self, config, fs=FileSaver()):
        self.config = config
        self.data_path = Path(config['raw_data_path'])
        self.results_path = config['results_path']
        self.detection_alg = config['detection_alg']
        self.fs = FileSaver()
        self.log = logging.getLogger(self.__class__.__name__)

    def load_data(self):
        """Load the data."""
        self.df = FileLoader().load_file(self.data_path)
        self.log.debug(f'Initial dataset shape {self.df.shape}')
        self.log.debug(f'Initial dataset sample:\n{self.df.sample(5)}')
        self.log.debug(f'Dataset loaded from {self.data_path}')
        return self.df

    def create_iso_df(self, dataset, prepared_data, anomalies, scores):
        """Create a dataframe from the prepared data."""
        time_stamp = dataset.data.timestamp.values
        sensor_n = self.config['sensor_n']
        try:
            df = pd.DataFrame(
                {
                    'timestamp': pd.to_datetime((time_stamp), unit='s'),
                    'unix': time_stamp,
                    f'sensor_{sensor_n}': prepared_data[:, sensor_n-1],
                    'anomaly': anomalies[f'sensor_{sensor_n}'],
                    'score': scores[f'sensor_{sensor_n}']
                }
            )
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            self.log.error(f'Error in pipeline: {e}', exc_info=True)

    def detect_level_shift(self, df):
        """Detect level shifts in the prepared data."""
        try:
            shift_detector = LevelShiftDetector(self.config, df)
            alarms = shift_detector.detect_shifts()
            self.log.info('Level shift detection completed')
            return alarms
        except Exception as e:
            self.log.error(f'Error in pipeline: {e}', exc_info=True)

    def generate_visualise(self, df, alarms):
        """Generate and visualise the results."""
        try:
            visualise = Visualiser(self.config, self.config['sensor_n'])
            visualise.get_visuals(df)
            if self.config['detection_alg'] == 'iso':
                visualise.apply_level_shifts(
                    alarms, shift_type=self.config['shift_alg'])
            else:
                pass
            plt.savefig(self.config['results_path_img'])
            return plt.show()
        except Exception as e:
            self.log.error(f'Error in pipeline: {e}', exc_info=True)

    def main(self):
        self.log.info(
            '========================== Beginning IsolationForest Pipeline =============================')

        self.log.info('Loading and Processing data...')
        raw_data = self.load_data()

        process = Processor(self.config)
        df = process.run_processing(raw_data)

        self.log.info('Preparing data for Isolation Forest...')
        dataset = SensorDataset(self.config, df)
        prepared_data = dataset.prepare_iso_data()
        dataset.log_summary()

        self.log.info('Detecting anomalies and apply Level Shift...')
        anomaly_model = IsolationForestAD(self.config)
        anomalies, scores = anomaly_model.detect_anomalies(prepared_data)

        df = self.create_iso_df(dataset, prepared_data, anomalies, scores)

        alarms = self.detect_level_shift(df)

        self.log.info('Visualising the results...')
        self.generate_visualise(df, alarms)
        self.fs.save_file(anomalies, self.results_path)

        self.log.info(
            '========================== Completed IsolationForest Pipeline =============================')

    def test(self):
        self.log.info(
            '====================================================TEST')


if __name__ == "__main__":
    project_dir, config, setup_logs = setup_project_env()
    pipeline = DataPipeline(config)
    pipeline.main()
    # pipeline.test()
