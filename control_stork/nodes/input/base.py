import torch

from .. import CellGroup
from ... extratypes import *


class InputGroup(CellGroup):
    """A special group which is used to supply batched dense tensor input to the network via its feed_data function."""

    def __init__(self, shape: Union[int, Iterable], name: str = "Input") -> None:
        super(InputGroup, self).__init__(shape, name=name)

    def reset_state(self, batch_size: Optional[int] = 1) -> None:
        super().reset_state(batch_size)
        self.out = self.states["out"] = torch.zeros(self.int_shape, device=self.device, dtype=self.dtype)

    def feed_data(self, data: torch.Tensor) -> None:
        self.local_data = data.reshape((data.shape[:2] + self.shape)).to(self.device)
        self.counter = 0

    def forward(self) -> None:

        self.out = self.states["out"] = self.local_data[:, self.counter]
        self.counter += 1