"""
This class compiles and saves the PyTorch model into a TorchScript for enhanced performance
"""
import torch
import custom_logger

from typing import Union
from torch.jit import ScriptModule


class Model:

    def __init__(self, model: Union[torch.nn.Module, torch.jit.ScriptModule]):
        self.model = model
        self.logger = custom_logger.get_logger(self.__class__.__name__)

    def compile(self, example: torch.Tensor, optimize: bool = True) -> ScriptModule:
        # TODO: guardare torch.compile per ottimizzare le performance su python<3.11
        if optimize:
            try:
                model = torch.compile(self.model)
                return torch.jit.trace(model, example)
            except RuntimeError as err:
                self.logger.warning("Torch high-performance compilation was not successful: %s", err)
                return torch.jit.trace(self.model, example)
        else:
            return torch.jit.trace(self.model, example)
