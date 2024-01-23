# -*- coding: utf-8 -*-
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.data.make_dataset import SensorDataset
from src.models.iso_forest import IsolationForestAD
from src.models.level_shift import LevelShiftDetector
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
        self.fs = FileSaver()

    def form_initial_dataset(self):
        """
        Form the initial dataset from the raw data.
        Returns:
            dataset (SensorDataset): Dataset object.
            prepared_data (pd.DataFrame): Prepared data.
        """
        try:
            dataset = SensorDataset(
                self.data_path, self.config)

            prepared_data = dataset.prepare_iso_data()
            self.logger.info(f'Dataset loaded from {self.data_path}')
            return dataset, prepared_data
        except Exception as e:
            self.logger.error(f'Error in pipeline: {e}', exc_info=True)

    def detect_anomalies(self, prepared_data):
        """
        Detect anomalies in the prepared data.
        Args:
            prepared_data (pd.DataFrame): Prepared data.
        Returns:
            anomalies (pd.DataFrame): Anomalies detected.
            scores (pd.DataFrame): Scores for each data point.
        """
        try:
            anomaly_model = IsolationForestAD(self.config)

            anomalies, scores = anomaly_model.detect_anomalies(prepared_data)
            self.logger.info('Anomaly detection completed')
            return anomalies, scores
        except Exception as e:
            self.logger.error(f'Error in pipeline: {e}', exc_info=True)

    def create_df(self, dataset, prepared_data, anomalies, scores):
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
        try:
            df = pd.DataFrame(
                {
                    'timestamp': pd.to_datetime((dataset.data.timestamp.values), unit='s'),
                    'unix': dataset.data.timestamp.values,
                    f'sensor_{self.config['sensor_n']}': prepared_data[:, self.config['sensor_n']-1],
                    'anomaly': anomalies[f'sensor_{self.config['sensor_n']}'],
                    'score': scores[f'sensor_{self.config['sensor_n']}']
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
            visualise.apply_level_shifts(
                alarms, shift_type=self.config['shift_alg'])
            return plt.show()
        except Exception as e:
            self.logger.error(f'Error in pipeline: {e}', exc_info=True)

    def run(self):
        self.logger.info(
            '====================================================')
        self.logger.info('Beginning IsolationForest pipeline')
        try:
            dataset, prepared_data = self.form_initial_dataset()
            dataset.log_summary()
            anomalies, scores = self.detect_anomalies(prepared_data)
            df = self.create_df(dataset, prepared_data, anomalies, scores)
            alarms = self.detect_level_shift(df)
            self.generate_visualise(df, alarms)
            self.fs.save_file(anomalies, self.results_path)
            self.logger.info(
                f'Anomaly detection results saved to {self.results_path}')

        except Exception as e:
            self.logger.error(f'Error in pipeline: {e}', exc_info=True)
        self.logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    project_dir, config = setup_project_env()
    pipeline = DataPipeline(project_dir, config)
    pipeline.run()
