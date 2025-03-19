import torch

from .. import CellGroup
from ...extratypes import *


class InputGroup(CellGroup):
    """A special group which is used to supply batched dense tensor input to the network via its feed_data function."""

    def __init__(
        self,
        shape: Union[int, Iterable],
        name: str = "Input",
        input_scale: float = 1.0,
        learn_input_scale: bool = False,
        **kwargs
    ) -> None:
        super(InputGroup, self).__init__(shape, name=name, **kwargs)
        self.learn_input_scale = learn_input_scale
        self.set_scaling(input_scale)

    def set_scaling(self, input_scale: float = 1.0) -> None:
        input_scale = torch.tensor([input_scale] * self.nb_units, dtype=torch.float32)
        if self.learn_input_scale:
            self.log_input_scale_ = torch.nn.Parameter(torch.log(input_scale))
        else:
            self.register_buffer("input_scale_", input_scale)

    def reset_state(self, batch_size: Optional[int] = 1) -> None:
        super().reset_state(batch_size)
        self.out = self.states["out"] = torch.zeros(
            self.int_shape, device=self.device, dtype=self.dtype
        )

    def feed_data(self, data: torch.Tensor) -> None:
        self.local_data = data.reshape((data.shape[:2] + self.shape)).to(self.device)
        self.counter = 0

    def forward(self) -> None:

        self.out = self.states["out"] = (
            self.local_data[:, self.counter] * self.input_scale
        )
        self.counter += 1

    # define a property for scaling
    @property
    def input_scale(self) -> torch.Tensor:
        if self.learn_input_scale:
            return torch.exp(self.log_input_scale_)
        else:
            return self.input_scale_
