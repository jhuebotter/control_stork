import torch
import torch.nn as nn

from .extratypes import *


class NetworkNode(nn.Module):
    def __init__(
        self, name: Optional[str] = None, regularizers: Optional[Iterable] = None
    ) -> None:
        """Initialize base class

        Args:
            name: A string name for this class used in logs
            regularizers: A list of regularizers for this class
        """
        super(NetworkNode, self).__init__()
        if name is None:
            self.name = ""
        else:
            self.name = name

        if regularizers is None:
            self.regularizers = []
        else:
            self.regularizers = regularizers

    def set_name(self, name: str) -> None:
        self.name = name

    def configure(
        self, time_step: float, device: torch.device, dtype: torch.dtype
    ) -> None:
        # removed batch_size and nb_steps

        self.time_step = self.dt = time_step
        self.device = device
        self.dtype = dtype

    def remove_regularizers(self) -> None:
        self.regularizers = []

    def __repr__(self) -> str:
        return super().__repr__() + " " + self.name
