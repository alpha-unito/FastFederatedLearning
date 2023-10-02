"""
Wrapper class for a PyTorch model.

This class compiles the PyTorch model into a TorchScript object that can then be passed to the C/C++ backend.
The model can also be compiled to obtained enhanced computational performance if possible.
"""
import torch
import logging

from typing import Union, Dict, Optional
from torch.jit import ScriptModule
from python_interface.utils import utils
from python_interface.custom.custom_types import PathLike
from python_interface.custom.custom_exceptions import WronglySpecifiedArgumentException


class Model:
    """Wrapper for a PyTorch model"""

    # TODO: make torchscript check automatic
    def __init__(self, model: Union[torch.nn.Module, torch.jit.ScriptModule], example: Optional[torch.Tensor] = None,
                 is_torchscript: bool = False, optimize: bool = True, torchscript_path: Optional[PathLike] = None):
        """Class that wraps the PyTorch model provided by the user and translates it into a TorchScript model.

        :param model: a PyTorch Model.
        :type model: a torch.nn.Module or a torch.jit.ScriptModule
        :param example: tensor with the same size as the model input.
        :type example: torch.Tensor
        :param is_torchscript: indicates if the provided model is already in a torchscript format.
        :type is_torchscript: bool
        :param optimize: if to compile the PyTorch model for enhanced computational performance.
        :type optimize: bool
        :param torchscript_path: path of the TorchScript model.
        :type torchscript_path: Optional[PathLike]
        """
        self.logger: logging.Logger = utils.get_logger(self.__class__.__name__)
        self.model: Union[torch.nn.Module, torch.jit.ScriptModule] = model
        self.example: Optional[torch.Tensor] = example
        self.is_torchscript: bool = is_torchscript
        self.optimize: bool = optimize
        self.torchscript_path: Optional[PathLike] = None

        try:
            self.check_torchscript_example()
        except WronglySpecifiedArgumentException as e:
            self.logger.critical(e.message)

        self.set_torchscript_path(torchscript_path)

        self.logger.info("Pytorch model created successfully.")

    def get_torchscript_path(self) -> Optional[PathLike]:
        """Get the TorchScript model path.

        :return: the TorchScript model path
        :rtype: PathLike
        """
        return self.torchscript_path

    def set_torchscript_path(self, torchscript_path: Optional[PathLike]):
        """Set the TorchScript model path.

        :param torchscript_path: TorchScript model path.
        :type torchscript_path: PathLike
        """
        utils.check_and_create_path(torchscript_path, "torchscript_path", self.logger)
        self.logger.info("Setting the TorchScript model path to %s", torchscript_path)
        self.torchscript_path: PathLike = torchscript_path

    def compile(self) -> ScriptModule:
        """Compilation of a PyTorch model into a TorchScript object.

        :return: TorchScript version of the PyTorch model.
        :rtype: ScriptModule
        """
        if self.is_torchscript:
            self.logger.info("The model is already in the TorchScript format, skipping its compilation.")
            return self.model
        else:
            if self.optimize:
                try:
                    self.logger.info("Compiling PyTorch model for improved computational performance...")
                    return torch.jit.trace(torch.compile(self.model), self.example)  # TODO: To be tested
                except RuntimeError as err:
                    self.logger.warning("Torch high-performance compilation was not successful: %s", err)
            self.logger.info("Compiling PyTorch model into TorchScript...")
            return torch.jit.trace(self.model, self.example)

    def check_torchscript_example(self):
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
