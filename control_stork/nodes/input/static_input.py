from . import InputGroup

from ...extratypes import *

class StaticInputGroup(InputGroup):
    """A special group which is used to supply batched dense tensor input to the network via its feed_data function."""

    def __init__(self, shape: Union[int, Iterable], scale: float = 1.0, name: str = "Input") -> None:
        super(StaticInputGroup, self).__init__(shape, name=name)
        self.scale = scale

    def forward(self):
        self.out = self.states["out"] = self.scale * self.local_data
