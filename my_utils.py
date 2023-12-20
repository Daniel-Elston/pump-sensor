# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
from pathlib import Path

import dotenv
import pyarrow.parquet as pq
import yaml


def load_config(config_path, encoding='utf-8'):
    """
    Load a YAML configuration file.
    """
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def setup_environment():
    """
    Set up the environment.
    """
    project_dir = Path(__file__).resolve().parents[0]
    dotenv.load_dotenv(os.path.join(project_dir, '.env'))
    return project_dir


def load_from_parquet(file_path):
    """
    Load a DataFrame from a Parquet file.

    Args:
        file_path (str): Path to the Parquet file.
    Returns:
        DataFrame: DataFrame containing the data.
    """
    df = pq.read_table(file_path).to_pandas()
    return df


def save_to_parquet(df, file_path):
    """
    Save a DataFrame to a Parquet file.

    Args:
        df (DataFrame): DataFrame to save.
        file_path (str): Path to the output Parquet file.
    """
    df.to_parquet(file_path, engine='pyarrow')


def save_to_json(data, file_path):
    """
    Save a dictionary to a JSON file.

    Args:
        data (dict): Dictionary to save.
        file_path (str): Path to the output JSON file.
    """
    with open(file_path, 'w') as file:
        json.dump(data, file)
    print('Anomaly detection results saved to results/iso1.json')
