"""
Wrapper class for a PyTorch model.

This class compiles the PyTorch model into a TorchScript object that can then be passed to the C/C++ backend.
The model can also be compiled to obtained enhanced computational performance if possible.
"""
import logging
import os
from typing import Union, Optional, NoReturn

import torch
from torch.jit import ScriptModule

from python_interface.custom.custom_exceptions import WronglySpecifiedArgumentException
from python_interface.custom.custom_types import PathLike
from python_interface.utils.utils import get_logger


class Model:
    """Wrapper for a PyTorch model"""

    def __init__(self, model: Optional[Union[torch.nn.Module, ScriptModule]] = None,
                 example: Optional[torch.Tensor] = None, optimize: bool = False):
        """Class that wraps the PyTorch model provided by the user and translates it into a TorchScript model.

        :param model: a PyTorch Model.
        :type model: a torch.nn.Module or a torch.jit.ScriptModule
        :param example: tensor with the same size as the model input.
        :type example: torch.Tensor
        :param optimize: if to compile the PyTorch model for enhanced computational performance.
        :type optimize: bool
        """
        self.logger: logging.Logger = get_logger(self.__class__.__name__)

        self.model: Optional[Union[torch.nn.Module, ScriptModule]] = model
        self.example: Optional[torch.Tensor] = example
        self.optimize: bool = optimize

        # If the model is not provided, we suppose that a TorchScript file is available
        self.is_torchscript: bool = isinstance(self.model, ScriptModule) or model is None
        self.exists: bool = False

        try:
            self.check_torchscript_example()
        except WronglySpecifiedArgumentException as e:
            self.logger.critical("Model parameters specification is incoherent: %s", e)

        self.logger.info("Pytorch model created successfully.")

    def check_torchscript_example(self) -> NoReturn | WronglySpecifiedArgumentException:
        """Utility method for checking that an example is specified only if the model is not already compiled,
        and viceversa.

        :raises: WronglySpecifiedArgumentException
        """
        if self.is_torchscript:
            if self.example is not None:
                raise WronglySpecifiedArgumentException(
                    "An example tensor has been specified even if the provided model is already a TorchScprit object.")
        else:
            if self.example is None:
                raise WronglySpecifiedArgumentException(
                    "An example tensor has not been specified even if the provided model is not a TorchScprit object.")

    def already_exists(self) -> bool:
        """Returns True if the TorchScript file exists already.

        :return: True if the TorchScript file exists already.
        :rtype: bool
        """
        return self.exists

    def compile(self) -> ScriptModule:
        """Compilation of a PyTorch model into a TorchScript object.

        :return: TorchScript version of the PyTorch model.
        :rtype: ScriptModule
        """
        if not self.is_torchscript:

            if self.optimize:
                try:
                    self.logger.info("Compiling PyTorch model for improved computational performance...")
                    return torch.jit.trace(torch.compile(self.model), self.example)  # TODO: To be tested
                except RuntimeError as e:
                    self.logger.warning("Torch high-performance compilation was not successful: %s", e)
                    self.logger.warning("Rolling back to standard compilation.")

            self.logger.info("Compiling PyTorch model into TorchScript...")
            return torch.jit.trace(self.model, self.example)

    def check_torchscript_no_model(self, torchscript_path: PathLike) -> NoReturn | WronglySpecifiedArgumentException:
        """Utility method for checking if the model is already saved in a TorchScript format.

        :raises: WronglySpecifiedArgumentException
        """
        if self.is_torchscript and self.model is None:
            self.logger.debug("Checking if the TorchSCript path exists...")
            if os.path.exists(torchscript_path):
                self.logger.info("The specified model path is correct - model found.")
                self.exists = True
            else:
                raise WronglySpecifiedArgumentException("The specified model path is not correct - model not found.")
