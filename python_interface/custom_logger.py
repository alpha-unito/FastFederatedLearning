"""
Small utility class to uniform logging across all classes.

All the configuration for the loggers is contained in the logging.conf file.
A different logger is defined for each class, defined by the class own name.
This way multiple instances of the same class will obtain the same instance of the class logger.
when a new class is created, a new entry should be inserted in the logging.conf file.
"""
import logging
import logging.config

LOGGING_CONFIGURATION = "logging.conf"
"""Path to the logging configuration file"""


def get_logger(class_name: str = "root") -> logging.Logger:
    """Returns the logger associated to the specified class name.
        Multiple instances of the same class asking for a logger will obtain access to the same logger.

        :param class_name: a string reporting the class name asking for a logger.
        :type class_name: str

        :return: Logger object.
        :rtype: logging.Logger
        """
    logging.config.fileConfig(LOGGING_CONFIGURATION)
    return logging.getLogger(class_name)
