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
        output_scale: float = 1.0,
        apply_tanh: bool = False,
        initial_state: float = -1e-3,
        learn_weight_scale: bool = False,
        learn_output_scale: bool = False,
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
        self.out = None
        self.syn = None
        self.apply_tanh = apply_tanh  # Whether to use tanh activation
        self.learn_weight_scale = learn_weight_scale
        self.learn_output_scale = learn_output_scale

        if not self.apply_tanh and self.learn_output_scale:
            raise ValueError(
                "Output scaling is only used when applying tanh activation"
            )

        self.set_scaling(weight_scale, output_scale)

    def set_scaling(
        self,
        weight_scale: float = 1.0,
        output_scale: float = 1.0,
    ) -> None:

        # Convert initial scaling values to log-space
        weight_scale = torch.tensor([weight_scale] * self.nb_units, dtype=torch.float32)
        output_scale = torch.tensor([output_scale] * self.nb_units, dtype=torch.float32)

        # Separate learning options for weight and output scaling
        if self.learn_weight_scale:
            self.log_weight_scale_ = torch.nn.Parameter(torch.log(weight_scale))
        else:
            self.register_buffer("weight_scale_", weight_scale)

        if self.learn_output_scale and self.apply_tanh:
            self.log_out_scale_ = torch.nn.Parameter(torch.log(output_scale))
        else:
            self.register_buffer("output_scale_", output_scale)

    def configure(self, time_step, device, dtype) -> None:
        super().configure(time_step, device, dtype)

        # TODO: change the time constant handling and add learning possibility
        self.dcy_mem = float(torch.exp(-time_step / torch.tensor(self.tau_mem)))
        self.scl_mem = 1.0 - self.dcy_mem
        self.dcy_syn = float(torch.exp(-time_step / torch.tensor(self.tau_syn)))
        self.scl_syn = (
            1.0 - self.dcy_syn
        ) * self.weight_scale  # this seems to be currently unused and could be removed

    def reset_state(self, batch_size: int = 1) -> None:
        super().reset_state(batch_size)
        self.out = self.get_state_tensor("out", state=self.out, init=self.initial_state)
        self.syn = self.get_state_tensor("syn", state=self.syn)

    def forward(self) -> None:
        # synaptic & membrane dynamics
        new_syn = self.dcy_syn * self.syn + self.input * self.weight_scale
        new_mem = self.dcy_mem * self.out + self.scl_mem * self.syn

        self.mem = self.states["mem"] = new_mem
        if self.apply_tanh:
            self.out = self.states["out"] = torch.tanh(new_mem) * self.output_scale
        else:
            self.out = self.states["out"] = self.mem
        self.syn = self.states["syn"] = new_syn

    @property
    def output_scale(self) -> torch.Tensor:
        """Returns the learned or fixed output scaling"""
        if self.learn_output_scale:
            return torch.exp(self.log_out_scale_)
        return self.output_scale_

    @property
    def weight_scale(self) -> torch.Tensor:
        """Returns the learned or fixed weight scaling"""
        if self.learn_weight_scale:
            return torch.exp(self.log_weight_scale_)
        return self.weight_scale_


class FastReadoutGroup(ReadoutGroup):
    def __init__(
        self,
        shape: Union[int, Iterable],
        tau_mem: Union[float, torch.Tensor] = 10e-3,
        tau_syn: Union[float, torch.Tensor] = 5e-3,
        weight_scale: float = 1.0,
        output_scale: float = 1.0,
        apply_tanh: bool = False,
        initial_state: float = -1e-3,
        learn_weight_scale: bool = False,
        learn_output_scale: bool = False,
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
            output_scale,
            apply_tanh,
            initial_state,
            learn_weight_scale,
            learn_output_scale,
            stateful,
            name="Fast Readout" if name is None else name,
            store_sequences=store_sequences,
            **kwargs
        )

    def forward(self) -> None:
        # synaptic & membrane dynamics
        new_syn = self.dcy_syn * self.syn + self.input * self.weight_scale
        new_mem = self.dcy_mem * self.out + self.scl_mem * new_syn
        self.mem = self.states["mem"] = new_mem
        if self.apply_tanh:
            self.out = self.states["out"] = torch.tanh(new_mem) * self.output_scale
        else:
            self.out = self.states["out"] = new_mem
        self.syn = self.states["syn"] = new_syn


class DirectReadoutGroup(CellGroup):
    def __init__(
        self,
        shape: Union[int, Iterable],
        weight_scale: float = 1.0,
        output_scale: float = 1.0,
        learn_weight_scale: bool = False,
        learn_output_scale: bool = False,
        apply_tanh: bool = False,
        initial_state: float = -1e-3,
        stateful: bool = False,
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
        self.initial_state = initial_state
        self.apply_tanh = apply_tanh  # Whether to use tanh activation

        # Convert initial scaling values to log-space
        log_weight_scale = torch.log(
            torch.tensor([weight_scale] * self.nb_units, dtype=torch.float32)
        )
        log_out_scale = torch.log(
            torch.tensor([output_scale] * self.nb_units, dtype=torch.float32)
        )

        # Separate learning options for weight and output scaling
        if learn_weight_scale:
            self.log_weight_scale = torch.nn.Parameter(log_weight_scale)
        else:
            self.register_buffer("log_weight_scale", log_weight_scale)

        if learn_output_scale:
            self.log_out_scale = torch.nn.Parameter(log_out_scale)
        else:
            self.register_buffer("log_out_scale", log_out_scale)

        self.out = None
        self.syn = None

    def reset_state(self, batch_size: int = 1) -> None:
        super().reset_state(batch_size)
        self.out = self.get_state_tensor("out", state=self.out, init=self.initial_state)

    def forward(self) -> None:
        """Compute the output with optional scaling and tanh activation"""

        mem = self.input * self.weight_scale
        if self.apply_tanh:
            self.out = self.states["out"] = torch.tanh(mem) * self.output_scale
        else:
            self.out = self.states["out"] = mem
        self.mem = self.states["mem"] = mem

    @property
    def output_scale(self) -> torch.Tensor:
        """Returns the learned or fixed output scaling"""
        return torch.exp(self.log_out_scale)

    @property
    def weight_scale(self) -> torch.Tensor:
        """Returns the learned or fixed weight scaling"""
        return torch.exp(self.log_weight_scale)


class TimeAverageReadoutGroup(CellGroup):
    def __init__(
        self,
        shape: Union[int, Iterable],
        weight_scale: float = 1.0,
        output_scale: float = 1.0,
        learn_weight_scale: bool = False,
        learn_output_scale: bool = False,
        apply_tanh: bool = False,
        steps: int = 1,
        initial_state: float = -1e-3,
        stateful: bool = False,
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
        self.initial_state = initial_state
        self.apply_tanh = apply_tanh
        self.steps = steps

        # Convert initial scaling values to log-space
        log_weight_scale = torch.log(
            torch.tensor([weight_scale] * self.nb_units, dtype=torch.float32)
        )
        log_out_scale = torch.log(
            torch.tensor([output_scale] * self.nb_units, dtype=torch.float32)
        )

        # Separate learning options for weight and output scaling
        if learn_weight_scale:
            self.log_weight_scale = torch.nn.Parameter(log_weight_scale)
        else:
            self.register_buffer("log_weight_scale", log_weight_scale)

        if learn_output_scale:
            self.log_out_scale = torch.nn.Parameter(log_out_scale)
        else:
            self.register_buffer("log_out_scale", log_out_scale)

        self.out = None
        self.syn = None
        self.memory = None

    def reset_state(self, batch_size: int = 1) -> None:
        super().reset_state(batch_size)
        self.out = self.get_state_tensor("out", state=self.out, init=self.initial_state)
        self.memory = [self.out] * self.steps  # Initialize memory buffer

    def forward(self) -> None:
        """Compute the time-averaged output with optional scaling and tanh activation"""

        mem = self.input * self.weight_scale
        self.memory.pop(0)
        self.memory.append(mem)

        avg_mem = torch.mean(torch.stack(self.memory), dim=0)

        if self.apply_tanh:
            self.out = self.states["out"] = torch.tanh(avg_mem) * self.output_scale
        else:
            self.out = self.states["out"] = avg_mem

        self.mem = self.states["mem"] = avg_mem

    @property
    def output_scale(self) -> torch.Tensor:
        """Returns the learned or fixed output scaling"""
        return torch.exp(self.log_out_scale)

    @property
    def weight_scale(self) -> torch.Tensor:
        """Returns the learned or fixed weight scaling"""
        return torch.exp(self.log_weight_scale)
