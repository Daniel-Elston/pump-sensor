# -*- coding: utf-8 -*-
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from src.data.make_dataset import SensorDataset
from src.models.iso_forest import IsolationForestAD
from src.models.level_shift import LevelShiftDetector
from src.models.lstm import LSTMAD
from src.visualization.visualize import Visualiser
from utils.file_log import Logger
from utils.file_save import FileSaver
from utils.setup_env import setup_project_env
warnings.filterwarnings(action='ignore', category=FutureWarning)


class DataPipeline:
    def __init__(self, project_dir, config, fs=FileSaver()):
        self.config = config
        self.logger = Logger(
            'PipelineLog', f'{Path(__file__).stem}.log').get_logger()
        self.project_dir = project_dir
        self.data_path = config['data_path']
        self.results_path = config['results_path']
        self.detection_alg = config['detection_alg']
        self.fs = FileSaver()

    def form_initial_dataset(self):
        try:
            dataset = SensorDataset(self.data_path, self.config)

            if self.detection_alg == 'iso':
                prepared_data = dataset.prepare_iso_data()
                return dataset, prepared_data

            elif self.detection_alg == 'lstm':
                prepared_data = dataset.prepare_lstm_data()
                len_data = len(prepared_data)
                train_size = int(0.7 * len_data)
                test_size = len_data - train_size
                train_data, test_data = random_split(
                    dataset, [train_size, test_size])
                return dataset, prepared_data, train_data, test_data

            else:
                raise ValueError('Invalid detection algorithm.')
        except Exception as e:
            self.logger.error(f'Error in pipeline: {e}', exc_info=True)

        self.logger.info(f'Dataset loaded from {self.data_path}')

    def create_iso_df(self, dataset, prepared_data, anomalies, scores):
        """
        Create a dataframe from the prepared data.
        Args:
            dataset (SensorDataset): Dataset object.
            prepared_data (pd.DataFrame): Prepared data.
            anomalies (pd.DataFrame): Anomalies detected.
            scores (pd.DataFrame): Scores for each data point.
        Returns:
            df (pd.DataFrame): Dataframe containing the prepared data.
        """
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
            self.logger.error(f'Error in pipeline: {e}', exc_info=True)

    def create_lstm_df(self, dataset, prepared_data, anomalies):
        """
        Create a dataframe from the prepared data.
        Args:
            dataset (SensorDataset): Dataset object.
            prepared_data (pd.DataFrame): Prepared data.
            anomalies (pd.DataFrame): Anomalies detected.
            scores (pd.DataFrame): Scores for each data point.
        Returns:
            df (pd.DataFrame): Dataframe containing the prepared data.
        """
        truncated_timestamps = dataset.data.timestamp.values[:len(
            prepared_data)]
        truncated_data = dataset.data.values[:len(prepared_data)]

        try:
            df = pd.DataFrame(
                {
                    'timestamp': pd.to_datetime((truncated_timestamps), unit='s'),
                    'unix': truncated_timestamps,
                    f'sensor_{self.config['sensor_n']}': truncated_data[:, self.config['sensor_n']],
                    'anomaly': anomalies
                }
            )
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            self.logger.error(f'Error in pipeline: {e}', exc_info=True)

    def detect_level_shift(self, df):
        """
        Detect level shifts in the prepared data.
        Args:
            df (pd.DataFrame): Dataframe containing the prepared data.
        Returns:
            alarms (pd.DataFrame): Level shift alarms.
        """
        try:
            shift_detector = LevelShiftDetector(self.config, df)
            alarms = shift_detector.detect_shifts()
            self.logger.info('Level shift detection completed')
            return alarms
        except Exception as e:
            self.logger.error(f'Error in pipeline: {e}', exc_info=True)

    def generate_visualise(self, df, alarms):
        """
        Generate and visualise the results.
        Args:
            df (pd.DataFrame): Dataframe containing the prepared data.
            alarms (pd.DataFrame): Level shift alarms.
        """
        try:
            visualise = Visualiser(self.config, self.config['sensor_n'])
            visualise.get_visuals(df)
            if self.config['detection_alg'] == 'iso':
                visualise.apply_level_shifts(
                    alarms, shift_type=self.config['shift_alg'])
            else:
                pass
            return plt.show()
        except Exception as e:
            self.logger.error(f'Error in pipeline: {e}', exc_info=True)

    def run_iso(self):
        self.logger.info(
            '====================================================')
        self.logger.info('Beginning IsolationForest pipeline')
        try:
            dataset, prepared_data = self.form_initial_dataset()
            dataset.log_summary()

            anomaly_model = IsolationForestAD(self.config)
            anomalies, scores = anomaly_model.detect_anomalies(prepared_data)

            df = self.create_iso_df(dataset, prepared_data, anomalies, scores)
            alarms = self.detect_level_shift(df)
            self.generate_visualise(df, alarms)
            self.fs.save_file(anomalies, self.results_path)
            self.logger.info(
                f'Anomaly detection results saved to {self.results_path}')

        except Exception as e:
            self.logger.error(f'Error in pipeline: {e}', exc_info=True)
        self.logger.info("Pipeline completed successfully")

    def run_lstm(self):
        self.logger.info(
            '====================================================')
        self.logger.info('Beginning LSTM pipeline')
        try:
            dataset, prepared_data, train_data, test_data = self.form_initial_dataset()
            dataset.log_summary()

            train_dataloader = DataLoader(
                train_data, self.config['batch_size'])
            test_dataloader = DataLoader(test_data, self.config['batch_size'])

            model = LSTMAD(self.config)
            model.train_model(train_dataloader)

            train_anomalies = model.detect_anomalies(train_dataloader)
            test_anomalies = model.detect_anomalies(test_dataloader)
            anomalies = np.concatenate(
                [train_anomalies, test_anomalies], axis=0).tolist()

            df = self.create_lstm_df(dataset, prepared_data, anomalies)
            self.generate_visualise(df, anomalies)
            self.fs.save_file(anomalies, self.results_path)
            self.logger.info(
                f'Anomaly detection results saved to {self.results_path}')

        except Exception as e:
            self.logger.error(f'Error in pipeline: {e}', exc_info=True)
        self.logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    project_dir, config = setup_project_env()
    pipeline = DataPipeline(project_dir, config)

    if config['detection_alg'] == 'iso':
        pipeline.run_iso()
    elif config['detection_alg'] == 'lstm':
        pipeline.run_lstm()
    else:
        raise ValueError('Invalid detection algorithm.')
