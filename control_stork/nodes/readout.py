import numpy as np
import torch

from . import CellGroup
from ..extratypes import *

# TODO: add docstrings


class ReadoutGroup(CellGroup):
    def __init__(
        self,
        shape: Union[int, Iterable],
        tau_mem: Union[float, torch.Tensor] = 10e-3,
        tau_syn: Union[float, torch.Tensor] = 5e-3,
        weight_scale: float = 1.0,
        initial_state: float = -1e-3,
        stateful: bool = False,  # ? what is this good for? Why is it not True?
        name: Optional[str] = None,
        store_sequences: Optional[Iterable] = ["out"],
        **kwargs
    ) -> None:
        super().__init__(
            shape,
            stateful=stateful,
            name="Readout" if name is None else name,
            store_sequences=store_sequences,
            **kwargs
        )
        self.tau_mem = tau_mem
        self.tau_syn = tau_syn
        self.store_output_seq = True
        self.initial_state = initial_state  # ? why is this not 0?
        self.weight_scale = weight_scale  # ? what is this good for?
        self.out = None
        self.syn = None

    def configure(self, time_step, device, dtype) -> None:
        super().configure(time_step, device, dtype)

        # TODO: change the time constant handling and add learning possibility
        self.dcy_mem = float(torch.exp(-time_step / torch.tensor(self.tau_mem)))
        self.scl_mem = 1.0 - self.dcy_mem
        self.dcy_syn = float(torch.exp(-time_step / torch.tensor(self.tau_syn)))
        self.scl_syn = (1.0 - self.dcy_syn) * self.weight_scale

    def reset_state(self, batch_size: int = 1) -> None:
        super().reset_state(batch_size)
        self.out = self.get_state_tensor("out", state=self.out, init=self.initial_state)
        self.syn = self.get_state_tensor("syn", state=self.syn)

    def forward(self) -> None:
        # synaptic & membrane dynamics
        new_syn = self.dcy_syn * self.syn + self.input
        new_mem = self.dcy_mem * self.out + self.scl_mem * self.syn

        self.out = self.states["out"] = self.states["mem"] = new_mem
        self.syn = self.states["syn"] = new_syn


class FastReadoutGroup(ReadoutGroup):
    def __init__(
        self,
        shape: Union[int, Iterable],
        tau_mem: Union[float, torch.Tensor] = 10e-3,
        tau_syn: Union[float, torch.Tensor] = 5e-3,
        weight_scale: float = 1.0,
        initial_state: float = -1e-3,
        stateful: bool = False,  # ? what is this good for? Why is it not True?
        name: Optional[str] = None,
        store_sequences: Optional[Iterable] = ["out"],
        **kwargs
    ) -> None:
        super().__init__(
            shape,
            tau_mem,
            tau_syn,
            weight_scale,
            initial_state,
            stateful,
            name="Fast Readout" if name is None else name,
            store_sequences=store_sequences,
            **kwargs
        )

    def forward(self) -> None:
        # synaptic & membrane dynamics
        new_syn = self.dcy_syn * self.syn + self.input
        new_mem = self.dcy_mem * self.out + self.scl_mem * new_syn

        self.out = self.states["out"] = new_mem
        self.syn = self.states["syn"] = new_syn


class DirectReadoutGroup(CellGroup):
    def __init__(
        self,
        shape: Union[int, Iterable],
        weight_scale: float = 1.0,
        initial_state: float = -1e-3,
        stateful: bool = False,  # ? what is this good for? Why is it not True?
        name: Optional[str] = None,
        store_sequences: Optional[Iterable] = ["out"],
        **kwargs
    ) -> None:
        super().__init__(
            shape,
            stateful=stateful,
            name="Direct Readout" if name is None else name,
            store_sequences=store_sequences,
            **kwargs
        )
        self.store_output_seq = True
        self.initial_state = initial_state  # ? why is this not 0?
        self.weight_scale = weight_scale  # ? what is this good for?
        self.out = None
        self.syn = None

    def reset_state(self, batch_size: int = 1) -> None:
        super().reset_state(batch_size)
        self.out = self.get_state_tensor("out", state=self.out, init=self.initial_state)

    def forward(self) -> None:
        self.out = self.states["out"] = self.input * self.weight_scale


class TimeAverageReadoutGroup(CellGroup):
    def __init__(
        self,
        shape: Union[int, Iterable],
        weight_scale: float = 1.0,
        steps: int = 1,
        initial_state: float = -1e-3,
        stateful: bool = False,  # ? what is this good for? Why is it not True?
        name: Optional[str] = None,
        store_sequences: Optional[Iterable] = ["out"],
        **kwargs
    ) -> None:
        super().__init__(
            shape,
            stateful=stateful,
            name="Time Average Readout" if name is None else name,
            store_sequences=store_sequences,
            **kwargs
        )
        self.store_output_seq = True
        self.initial_state = initial_state  # ? why is this not 0?
        self.weight_scale = weight_scale
        self.steps = steps
        self.out = None
        self.syn = None

    def reset_state(self, batch_size: int = 1) -> None:
        super().reset_state(batch_size)
        self.out = self.get_state_tensor("out", state=self.out, init=self.initial_state)
        self.memory = [self.out] * self.steps

    def forward(self) -> None:
        self.memory.pop(0)
        self.memory.append(self.input * self.weight_scale)
        self.out = self.states["out"] = torch.mean(torch.stack(self.memory), dim=0)
