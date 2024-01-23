# -*- coding: utf-8 -*-
from __future__ import annotations

from sklearn.ensemble import IsolationForest


class IsolationForestAD:
    def __init__(self, config):
        self.config = config
        self.contamination = config['contamination']

    def detect_anomalies(self, data):
        """
        Detect anomalies for each sensor.
        Args:
            data (np.ndarray): Numpy array of sensor data.
        Returns:
            dict: Dictionary with sensor names as keys and anomaly labels as values.
        """
        results = {}
        scores = {}
        num_sensors = data.shape[1]

        for sensor_idx in range(num_sensors):
            # Reshape for sklearn
            sensor_data = data[:, sensor_idx].reshape(-1, 1)

            clf = IsolationForest(
                random_state=42, contamination=self.contamination)
            clf.fit(sensor_data)
            predictions = clf.predict(sensor_data)

            anomaly_score = clf.decision_function(sensor_data.reshape(-1, 1))

            sensor_name = f'sensor_{sensor_idx + 1}'
            results[sensor_name] = predictions.tolist()
            scores[sensor_name] = anomaly_score.tolist()
        return results, scores
