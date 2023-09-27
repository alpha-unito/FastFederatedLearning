"""Custom logging.Formatter to obtain coloured output"""
import logging

from typing import Literal, Optional


class CustomFormatter(logging.Formatter):
    """Logging colored formatter, adapted from https://stackoverflow.com/a/56944256/3638629"""

    grey = '\x1b[38;21m'
    blue = '\x1b[38;5;39m'
    yellow = '\x1b[38;5;226m'
    red = '\x1b[38;5;196m'
    bold_red = '\x1b[31;1m'
    reset = '\x1b[0m'

    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None, style: Literal["%", "{", "$"] = '%',
                 validate: bool = True):
        """Creation of the CustomFormatter.

        :param fmt: format of the logged message.
        :type fmt: Optional[str]
        :param datefmt: format of the logged date.
        :type datefmt: Optional[str]
        :param style: style of the formatted logged messages.
        :type style: str
        :param validate: validate the logging style.
        :type validate: bool
        """
        super().__init__(fmt, datefmt, style, validate)

        self.fmt = "%(asctime)s | %(name)16s | %(levelname)8s | %(message)s"  # TODO: get this format from logging.conf
        self.FORMATS = {
            logging.INFO: self.grey + self.fmt + self.reset,
            logging.DEBUG: self.blue + self.fmt + self.reset,
            logging.WARNING: self.yellow + self.fmt + self.reset,
            logging.ERROR: self.red + self.fmt + self.reset,
            logging.CRITICAL: self.bold_red + self.fmt + self.reset
        }

    def format(self, record):
        """Format the log.

        :param record: log record.
        :return: formatted log record.
        """
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
