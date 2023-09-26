"""
This class compiles the PyTorch model into a TorchScript object that can then be passed to the C/C++ backend.
The model can also be compiled to obtained enhanced computational performance if possible.
"""
import torch
import logging
import utils

from typing import Union
from torch.jit import ScriptModule


class Model:
    """Wrapper for a PyTorch model"""

    def __init__(self, model: Union[torch.nn.Module, torch.jit.ScriptModule]):
        """Class that wraps the PyTorch model provided by the user and translates it into a TorchScript model.

        :param model: a PyTorch Model.
        :type model: a torch.nn.Module or a torch.jit.ScriptModule
        """
        self.logger: logging.Logger = utils.get_logger(self.__class__.__name__)
        self.model: Union[torch.nn.Module, torch.jit.ScriptModule] = model

        self.logger.info("Pytorch model created successfully.")

    def compile(self, example: torch.Tensor, optimize: bool = True) -> ScriptModule:
        """Compilation of a PyTorch model into a TorchScript object.

        :param example: tensor with the same size as the model input.
        :type example: torch.Tensor
        :param optimize: if to compile the PyTorch model for enhanced computational performance.
        :type optimize: bool

        :return: TorchScript version of the PyTorch model.
        :rtype: ScriptModule
        """
        if optimize:
            try:
                self.logger.info("Compiling PyTorch model for improved computational performance...")
                return torch.jit.trace(torch.compile(self.model), example)  # TODO: To be tested
            except RuntimeError as err:
                self.logger.warning("Torch high-performance compilation was not successful: %s", err)
        self.logger.info("Compiling PyTorch model into TorchScript...")
        return torch.jit.trace(self.model, example)
