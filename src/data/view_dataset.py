# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import dotenv
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


project_dir = Path(__file__).resolve().parents[2]
dotenv.load_dotenv(os.path.join(project_dir, '.env'))
sys.path.append(str(project_dir))


class SensorDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Initialize the dataset object.

        Args:
            csv_file (str): Path to the CSV file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        # Load the data, transform and select
        self.sensor_data = pd.read_csv(csv_file)

        # Separate timestamps for display/plotting
        self.timestamps = self.sensor_data.iloc[:, 1]

        self.transform = transform
        self.data = self.sensor_data.iloc[:, 2:-1]  # Just sensor data

    def __len__(self):
        """
        Return the total number of samples in the dataset. Each sample is a sequence of data points.
        """
        return len(self.sensor_data)

    def __getitem__(self, idx):
        """
        Retrieve a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            torch.Tensor: A tensor representing a sequence of sensor readings.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sequence = self.data.iloc[idx]

        if self.transform:
            sequence = self.transform(sequence)

        # Convert the sequence to a tensor
        return torch.tensor(sequence.values, dtype=torch.float32), self.timestamps.iloc[idx]


def main(csv_file, index=None):

    path_to_csv_file = os.path.join(project_dir, csv_file)
    dataset = SensorDataset(csv_file=path_to_csv_file)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    if index is not None:
        # Retrieve a single item using __getitem__
        data, timestamp = dataset[index]
        print(f"Data at index {index}: {data}")
        print(f"Timestamp: {timestamp}")
    else:
        for i, (batch, timestamps) in enumerate(dataloader):
            print(f"Batch {i+1}:")
            print(batch, timestamps)
            if i == 2:
                break


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(description="View sensor data.")
    parser.add_argument('csv_file', type=str, help='Path to the CSV file')
    parser.add_argument('--index', type=int,
                        help='Index of the item to retrieve', required=False)
    args = parser.parse_args()

    main(csv_file=args.csv_file, index=args.index)

# python view_dataset.py --index 4 ...data\raw\sensor.csv
