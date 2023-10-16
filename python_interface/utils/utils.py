"""
Small utility module.
"""
import logging
import logging.config
import os
from typing import Any, get_args, NoReturn

from python_interface.custom.custom_exceptions import MutuallyExclusiveArgumentsException
from python_interface.custom.custom_types import PathLike
from python_interface.utils.constants import LOGGING_CONFIGURATION, VERSION

LOGGER_NUMBER: int = 0
"""Number of different loggers currently instantiated"""


def get_logger(class_name: str = "root") -> logging.Logger:
    """Returns the logger associated to the specified class name.
        All the configuration for the loggers is contained in the logging.conf file.
        A different logger is defined for each class, defined by the class own name.
        This way multiple instances of the same class will obtain the same instance of the class logger.
        when a new class is created, a new entry should be inserted in the logging.conf file.

    :param class_name: a string reporting the class name asking for a logger.
    :type class_name: str

    :return: Logger object.
    :rtype: logging.Logger
    """
    global LOGGER_NUMBER
    LOGGER_NUMBER += 1

    logging.config.fileConfig(LOGGING_CONFIGURATION)
    if LOGGER_NUMBER == 1:
        logger: logging.Logger = logging.getLogger("root")
        logger.info("FastFederatedLearning (FastFL) - %s - Starting the execution...", VERSION)
    return logging.getLogger(class_name)


def check_and_create_path(path: PathLike, target: str = "", logger: logging.Logger = logging):  # TODO: eccezioni?
    """Check if the provided path already exists; otherwise it will be created.

    :param path: path to check.
    :type path: PathLike
    :param target: name of the entity to which the path refers to.
    :type target: str
    :param logger: logger for reporting the operatinal status and debug.
    :type logger: logging.Logger
    """
    dirname = os.path.dirname(path)
    logger.debug("Attempting to create %s at path: %s...", target, path)
    if os.path.exists(dirname):
        logger.info("Path: %s already existing.", path)
    else:
        os.makedirs(dirname, exist_ok=True)
        logger.info("Created path: %s.", path)


def check_mutually_exclusive_args(arg_1: Any, arg_2: Any,
                                  logger: logging.Logger = logging) -> NoReturn | MutuallyExclusiveArgumentsException:
    """Check if the specified arguments are mutually exclusive.

    :param arg_1: first argument.
    :type arg_1: any
    :param arg_2: second argument.
    :type arg_2: any
    :param logger: logger for reporting the operatinal status and debug.
    :type logger: logging.Logger
    :raises: MutuallyExclusiveArgumentsException
    """
    if (arg_1 is not None and arg_2 is not None) or (arg_1 is None and arg_2 is None):
        logger.critical("Mutually exclusive arguments have been both specified: %s and %s.", arg_1, arg_2)
        raise MutuallyExclusiveArgumentsException("Mutually exclusive arguments.")


def check_var_in_literal(var: Any, literal: Any, logger: logging.Logger = logging) -> NoReturn | ValueError:
    """Check if the variable is in a list of accepted values.

    :param var: variable to check.
    :type var: any
    :param literal: reference values.
    :type literal: any
    :param logger: logger for reporting the operatinal status and debug.
    :type logger: logging.Logger
    :raises: ValueError
    """
    if var not in get_args(literal):
        logger.critical("Specified value is not allowed: %s not in %s.", var, literal)
        raise ValueError("Value " + str(var) + " not in " + str(literal))


def check_positive_int(var: int, threshold: int = 0, logger: logging.Logger = logging) -> NoReturn | ValueError:
    """Check if the variable is an integer greater than a threshold.

    :param var: variable to check.
    :type var: int
    :param threshold: threshold value.
    :type threshold: int
    :param logger: logger for reporting the operatinal status and debug.
    :type logger: logging.Logger
    :raises: ValueError
    """
    if var <= threshold:
        logger.critical("Specified value is positive %s is not greater than %s", var, threshold)
        raise ValueError("Value " + str(var) + " must be greater than" + str(threshold))
