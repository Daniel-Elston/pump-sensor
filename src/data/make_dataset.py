# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from pathlib import Path

import dotenv
import pandas as pd
import torch
from torch.utils.data import Dataset
# import argparse
# import sys

print(torch.__version__)  # PyTorch version = 2.2.0

project_dir = Path(__file__).resolve().parents[2]
dotenv_path = os.path.join(project_dir, '.env')
dotenv.load_dotenv(dotenv_path)


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

        # Separate timestamps
        self.timestamps = self.sensor_data.iloc[:, 1]

        self.transform = transform
        self.data = self.sensor_data.iloc[:, 2:-1]  # Select only sensor data

    def __len__(self):
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


def main():

    path_to_csv_file = os.path.join(project_dir, 'data/raw/sensor.csv')
    dataset = SensorDataset(csv_file=path_to_csv_file)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True)

    for i, (batch, timestamps) in enumerate(dataloader):
        print(f"Batch {i+1}:")
        print(batch, timestamps)

        # Break after printing a few batches
        if i == 2:
            break


if __name__ == "__main__":
    main()
