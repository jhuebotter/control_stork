import torch
import torch.nn as nn

import numpy as np

from . import CellGroup
from ..extratypes import *

# TODO: add docstrings


class FanOutGroup(CellGroup):
    def __init__(
        self,
        parent_group: CellGroup,
        fanout: int = 2,
        dim: int = 0,
        unsqueeze: bool = False,
    ) -> None:
        """Fan out output of parent group either by tiling along the neuron dimension or along a separate axis dimension.

        Args:
            parent_group (CellGroup): The parent group
            fanout (int): Factor to fan out
            dim (int, optional): The dimension along which to repeat the inputs.
            unsqueeze (bool): If true add extra dimension for fanout
        """
        super(FanOutGroup, self).__init__(parent_group.shape)
        self.dim_with_batch = dim + 1  # We add one for the batch dimension
        self.parent_group = parent_group
        self.fanout = fanout
        self.unsqueeze = unsqueeze

        # Update shape
        shape = list(parent_group.shape)
        if unsqueeze:
            shape = shape[:dim] + [fanout] + shape[dim:]
        else:
            shape[dim] = fanout * shape[dim]

        self.shape = tuple(shape)
        self.nb_units = int(np.prod(self.shape))

    def configure(
        self, time_step: float, device: torch.device, dtype: torch.dtype
    ) -> None:
        super().configure(time_step, device, dtype)

    def forward(self) -> None:
        inputs = self.parent_group.out
        if self.unsqueeze:
            inputs = torch.unsqueeze(inputs, self.dim_with_batch)
        self.out = torch.repeat_interleave(inputs, self.fanout, self.dim_with_batch)


class TorchOp(CellGroup):
    """Apply an arbitrary torch op to the output of a dedicated parent neuron group at every time step."""

    def __init__(
        self,
        parent_group: CellGroup,
        operation: Callable,
        shape: Optional[Union[int, Iterable]] = None,
        **kwargs
    ) -> None:
        if shape is None:
            shape = parent_group.shape
        super(TorchOp, self).__init__(shape, **kwargs)
        self.parent_group = parent_group
        self.op = operation

    def add_to_state(self, target: str, x: torch.Tensor) -> None:
        """Just implement pass through to parent group."""
        self.parent_group.add_to_state(target, x)

    def reset_state(self, batch_size: int = 1) -> None:
        super().reset_state(batch_size)
        self.out = self.states["out"] = self.parent_group.out

    def forward(self) -> None:
        x = self.parent_group.out
        self.out = self.op(x)


class MaxPool1d(TorchOp):
    """Apply 1D MaxPooling to output of the parent neuron group."""

    def __init__(self, parent_group: CellGroup, kernel_size: int = 2, **kwargs) -> None:
        shape = parent_group.shape[:-1] + (parent_group.shape[-1] // kernel_size,)
        super(MaxPool1d, self).__init__(
            parent_group, operation=nn.MaxPool1d(kernel_size), shape=shape, **kwargs
        )


class MaxPool2d(TorchOp):
    """Apply 2D MaxPooling to output of the parent neuron group."""

    def __init__(self, parent_group: CellGroup, kernel_size: int = 2, **kwargs):
        shape = parent_group.shape[:-2] + (
            parent_group.shape[-2] // kernel_size,
            parent_group.shape[-1] // kernel_size,
        )
        super(MaxPool2d, self).__init__(
            parent_group, operation=nn.MaxPool2d(kernel_size), shape=shape, **kwargs
        )
