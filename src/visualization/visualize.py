# -*- coding: utf-8 -*-
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


class Visualiser:
    def __init__(self, config, sensor_n):
        self.config = config
        self.sensor_n = sensor_n

    def create_df(self, dataset, prepared_data, anomalies, scores):
        """
        Create a DataFrame from the sensor data, anomalies, and scores.

        Args:
            sensor_n (int): The sensor number to process.
            dataset (SensorDataset): The dataset containing the sensor data.
            prepared_data (np.array): The prepared sensor data.
            anomalies (np.array): Anomaly flags for the data.
            scores (np.array): Anomaly scores for the data.

        Returns:
            pd.DataFrame: A DataFrame with the processed data.
        """
        df = pd.DataFrame(
            {
                'timestamp': pd.to_datetime((dataset.data.timestamp.values), unit='s'),
                'unix': dataset.data.timestamp.values,
                f'sensor_{self.sensor_n}': prepared_data[:, self.sensor_n-1],
                'anomaly': anomalies[f'sensor_{self.sensor_n}'],
                'score': scores[f'sensor_{self.sensor_n}']
            }
        )
        df.set_index('timestamp', inplace=True)
        return df

    def get_visuals(self, df):
        """
        Generate visualizations for sensor data including anomalies.

        Args:
            sensor_n (int): The sensor number to visualize.
            df (pd.DataFrame): The DataFrame containing the sensor data.
            figsize (tuple): The size of the figure.
            normal_color (str): The color for normal data points.
            anomaly_color (str): The color for anomalies.

        Returns:
            matplotlib.figure.Figure: The figure object with the plot.
        """
        get_anomaly = df[df['anomaly'] == -1]
        fig, ax = plt.subplots(figsize=(18, 8))
        plt.plot(
            df[[f'sensor_{self.sensor_n}']],
            color='blue',
            label='normal',
            linewidth=0.5
        )
        plt.scatter(
            get_anomaly.index,
            get_anomaly[[f'sensor_{self.sensor_n}']],
            color='red',
            label='anomaly',
            s=10
        )

    def apply_level_shifts(self, alarms, shift_type):
        """
        Apply level shifts to the plot.

        Args:
            alarms (pd.Series): The alarms to apply.
            shift_type (str): The type of level shift to apply.
        """
        if shift_type == 'ruptures':
            for i in range(len(alarms)):
                plt.axvline(
                    x=alarms.index[i],
                    color='green',
                    linewidth=0.5
                )
                if i % 2 == 0:
                    plt.axvspan(
                        alarms.index[i],
                        alarms.index[i+1],
                        facecolor='r',
                        alpha=0.1
                    )

        elif shift_type == 'adtk':
            for i in range(len(alarms)):
                plt.axvline(
                    x=alarms.index[i],
                    color='green',
                    linewidth=1,
                    alpha=0.1
                )
