import torch

from .. import CellGroup
from ...extratypes import *


class InputGroup(CellGroup):
    """A special group which is used to supply batched dense tensor input to the network via its feed_data function."""

    def __init__(
        self,
        shape: Union[int, Iterable],
        name: str = "Input",
        scaling: float = 1.0,
        learn_scaling: bool = False,
        **kwargs
    ) -> None:
        super(InputGroup, self).__init__(shape, name=name, **kwargs)
        log_scaling = torch.log(torch.tensor(scaling, dtype=torch.float32))
        if learn_scaling:
            self.log_scaling = torch.nn.Parameter(
                torch.tensor(log_scaling, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "log_scaling", torch.tensor(log_scaling, dtype=torch.float32)
            )

    def reset_state(self, batch_size: Optional[int] = 1) -> None:
        super().reset_state(batch_size)
        self.out = self.states["out"] = torch.zeros(
            self.int_shape, device=self.device, dtype=self.dtype
        )

    def feed_data(self, data: torch.Tensor) -> None:
        self.local_data = data.reshape((data.shape[:2] + self.shape)).to(self.device)
        self.counter = 0

    def forward(self) -> None:

        self.out = self.states["out"] = self.local_data[:, self.counter] * self.scaling
        self.counter += 1

    # define a property for scaling
    @property
    def scaling(self) -> torch.Tensor:
        return torch.exp(self.log_scaling)
