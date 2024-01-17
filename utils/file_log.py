from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler


class Logger:
    def __init__(
            self, name, log_file, level=logging.INFO):
        """
        Initialize the Logger.

        :param name: Name of the logger.
        :param log_file: File path for the log file.
        :param level: Logging level, e.g., logging.INFO, logging.DEBUG.
        """
        # Create a logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Create handlers
        file_handler = RotatingFileHandler(
            f'log/{log_file}', maxBytes=1000000, backupCount=5)
        console_handler = logging.StreamHandler()

        # Create formatters and add to handlers
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_formatter = logging.Formatter(
            '%(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(console_formatter)

        # Add handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        # Log separator
        # self.logger.info(separator)

    def get_logger(self):
        """
        Returns the configured logger.
        """
        return self.logger
