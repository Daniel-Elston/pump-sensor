from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler

from utils.setup_env import setup_project_env
project_dir, config = setup_project_env()


class Logger:
    def __init__(
            self, name, log_file):
        """
        Initialize the Logger.

        :param name: Name of the logger.
        :param log_file: File path for the log file.
        :param level: Logging level, e.g., logging.INFO, logging.DEBUG.
        """
        # Create a logger
        level = config['logging']['level']
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level))

        # Create handlers
        file_handler = RotatingFileHandler(
            os.path.join(project_dir, f'log/{log_file}'), maxBytes=1000000, backupCount=5)
        console_handler = logging.StreamHandler()

        # Set levels manually
        console_handler.setLevel(logging.INFO)

        # Create formatters and add to handlers
        file_formatter = logging.Formatter(
            '%(pathname)s - %(asctime)s - %(levelname)s - %(filename)s'
            ' - %(lineno)d - %(module)s - %(funcName)s - %(name)s - %(message)s')
        console_formatter = logging.Formatter(
            '%(name)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s')
        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(console_formatter)

        # Add handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        # generate log separator

    def get_logger(self):
        """
        Returns the configured logger.
        """
        return self.logger
