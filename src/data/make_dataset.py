# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from pathlib import Path

import dotenv
import pandas as pd

# project_dir = str(Path.cwd().parent.parent)
project_dir = Path(__file__).resolve().parents[2]

dotenv_path = os.path.join(project_dir, '.env')
dotenv.load_dotenv(dotenv_path)


def load_csv(file_path):
    df = pd.read_csv(file_path)
    return df


def save_as_parq(df, save_path):
    df.to_parquet(save_path)


def load_parquet(file_path):
    df = pd.read_parquet(file_path)
    return df


def main():

    csv_file_path = 'C:/Users/delst/workspace/pump-sensor/data/raw/sensor.csv'
    parquet_file_path = 'C:/Users/delst/workspace/pump-sensor/data/sdo/sensor.parq'

    # Load data from CSV
    df = load_csv(csv_file_path)
    sample = df.iloc[0:10000, :]  # 1 day sample data
    sample.to_csv(
        'C:/Users/delst/workspace/pump-sensor/data/sample/sensor_sample.csv')

    # Save as Parquet
    save_as_parq(df, parquet_file_path)

    # Load from Parquet
    df_parquet = load_parquet(parquet_file_path)

    # Display the DataFrame
    print(df_parquet)


if __name__ == "__main__":
    main()
