"""
This class compiles and saves the PyTorch model into a TorchScript for enhanced performance
"""
import torch

from typing import Union
from torch.jit import ScriptModule


class Model:

    def __init__(self, model: Union[torch.nn.Module, torch.jit.ScriptModule]):
        self.model = model  # TODO: guardare torch.compile per ottimizzare le performance

    def compile(self, example: torch.Tensor) -> ScriptModule:
        return torch.jit.trace(self.model, example)
