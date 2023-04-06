import numpy as np
import torch

from . import InputGroup
from ... extratypes import *

class RasInputGroup(InputGroup):
    """ Like InputGroup but eats ras format instead of dense tensors. """

    def __init__(self, shape: Union[int, Iterable], name: str = "Input") -> None:
        super(RasInputGroup, self).__init__(shape, name=name)

    def forward(self) -> None:
        tmp = torch.zeros(self.int_shape, dtype=self.dtype)
        for bi, dat in enumerate(self.local_data):
            times, units = dat
            idx = np.array(units[times == self.clk], dtype=np.int)
            tmp[bi, idx] = 1.0
        self.out = tmp.to(device=self.device)