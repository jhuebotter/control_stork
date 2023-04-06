import torch

from . import InputGroup
from ... extratypes import *

class SparseInputGroup(InputGroup):
    """ Like InputGroup but eats sparse tensors instead of dense ones. """

    def __init__(self, shape: Union[int, Iterable], name: str = "Input") -> None:
        super(SparseInputGroup, self).__init__(shape, name=name)

    def forward(self):
        i, v = self.local_data[self.clk]
        tmp = torch.sparse.FloatTensor(i, v, torch.Size(self.int_shape))
        # print(self.int_shape)
        self.out = tmp.to(self.device)