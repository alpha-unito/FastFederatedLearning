"""
Wrapper class for a generic dataset.

This class also offers methods for data splitting.
"""
import logging

from python_interface.utils import utils
from python_interface.custom.custom_types import PathLike


class Dataset():
    """Wrapper for a generic dataset"""

    def __init__(self, data_path: PathLike):
        """Class that wrapsa generic dataset provided by the user and allows to split it across different clients.

        :param data_path: a generic dataset.
        :type data_path:
        """
        self.logger: logging.Logger = utils.get_logger(self.__class__.__name__)
        self.data_path: PathLike = data_path

        self.logger.info("Dataset loaded successfully.")

    def set_data_path(self, data_path: PathLike):
        """Set the dataset path.

        :param data_path: dataset path.
        :type data_path: PathLike
        """

        utils.check_and_create_path(data_path, "data_path", self.logger)
        self.logger.info("Setting the dataset path to %s", data_path)
        self.data_path: PathLike = data_path

    def get_data_path(self) -> PathLike:
        """Get the dataset path.

        :return: the dataset path
        :rtype: PathLike
        """
        return self.data_path
