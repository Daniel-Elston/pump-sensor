# -*- coding: utf-8 -*-
from __future__ import annotations

import torch
from sklearn.ensemble import IsolationForest
from torch.utils.data import DataLoader


class IsolationForestAD:
    def __init__(self, dataset, config, batch_size=1, contamination=0.2):
        self.dataset = dataset
        self.config = config
        self.dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False)
        self.contamination = contamination

    def prepare_data(self):
        """
        Prepare the data for anomaly detection.
        """
        sensor_data_list = []

        for batch in self.dataloader:
            sensor_data = batch[:, 1:]  # Exclude the timestamp column
            sensor_data_list.append(sensor_data)

        # Convert list of tensors to a single numpy array
        sensor_data_np = torch.cat(sensor_data_list).numpy()
        return sensor_data_np

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


# def main():
#     data_path = os.path.join(project_dir, 'data/sdo/sensor.parq')
#     results_path = os.path.join(project_dir, 'results/iso1.json')

#     config_path = os.path.join(project_dir, 'my_config.yaml')
#     config = load_config(config_path)

#     dataset = SensorDataset(
#         data_path, config, time_window=config['time_window'])

#     anomaly_model = IsolationForestAD(
#         dataset, config, contamination=config['contamination'])
#     prepared_data = anomaly_model.prepare_data()

#     anomalies = anomaly_model.detect_anomalies(prepared_data)

#     with open(results_path, 'w') as file:
#         json.dump(anomalies, file)
#     print('Anomaly detection results saved to results/iso1.json')


# if __name__ == "__main__":
#     main()
