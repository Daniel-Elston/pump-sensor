# -*- coding: utf-8 -*-
from __future__ import annotations

import matplotlib.pyplot as plt


class Visualiser:
    def __init__(self, config, sensor_n):
        self.config = config
        self.sensor_n = sensor_n

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
        # get_anomaly = df[df['anomaly'] == 1] change for lstm

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
            s=5
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
                    color='r',
                    linewidth=1,
                    alpha=0.1
                )
