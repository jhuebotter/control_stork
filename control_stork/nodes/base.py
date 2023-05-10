import torch
import torch.nn as nn

import numpy as np

from .. import core
from ..extratypes import *

# TODO: add docstrings


class CellGroup(core.NetworkNode):
    """
    Base class from which all neurons are derived.

    """

    def __init__(
        self,
        shape: Union[int, Iterable],
        store_sequences: Optional[Iterable] = None,
        name: Optional[str] = None,
        regularizers: Optional[Iterable] = None,
        dropout_p: float = 0.0,
        stateful: bool = False,
        **kwargs
    ) -> None:
        super(CellGroup, self).__init__(name, regularizers)
        if type(shape) == int:
            self.shape = (shape,)
        else:
            self.shape = tuple(shape)
        self.nb_units = int(np.prod(self.shape))
        self.states = {}
        self.store_state_sequences = []  # used to contain "out"
        # this was changed to avoid memory overflow for processing long sequences
        if store_sequences is not None:
            self.store_state_sequences.extend(store_sequences)
            self.store_state_sequences = list(set(self.store_state_sequences))
        self.stored_sequences_ = {}
        self.default_target = "input"
        self.stateful = stateful
        if dropout_p:
            self.dropout = nn.Dropout(dropout_p)
        else:
            self.dropout = None
        self.clk = 0

    def configure(
        self, time_step: float, device: torch.device, dtype: torch.dtype
    ) -> None:
        super().configure(time_step, device, dtype)
        self.reset_state()

    def get_regularizer_loss(self) -> torch.Tensor:
        reg_loss = torch.tensor(0.0, device=self.device)
        for reg in self.regularizers:
            reg_loss += reg(self)
        return reg_loss

    def set_state_tensor(self, key: str, state: torch.Tensor) -> None:
        self.states[key] = state

    def prepare_state_tensor_(
        self,
        state: Optional[torch.Tensor] = None,
        init: float = 0.0,
        shape: Optional[Tuple] = None,
    ) -> torch.Tensor:
        """Prepares a state tensor by either initializing it or copying the previous one.

        Args:
            state (tensor): The previous state tensor if one exists
            init (float): Numerical value to init tensor with
            shape (None or tuple): Shape of the state. Assuming a single value if none.

        Returns:
            A tensor with dimensions current_batch_size x neuronal_shape x shape
        """

        if self.stateful and state is not None and state.size() == self.int_shape:
            new_state = state.detach()
        else:
            if shape is None:
                full_shape = self.int_shape
            else:
                full_shape = self.int_shape + shape

            if init:
                new_state = init * torch.ones(
                    full_shape, device=self.device, dtype=self.dtype
                )
            else:
                new_state = torch.zeros(
                    full_shape, device=self.device, dtype=self.dtype
                )

        return new_state

    def get_state_tensor(
        self,
        key: str,
        state: Optional[torch.Tensor] = None,
        init: float = 0.0,
        shape: Optional[Tuple] = None,
    ) -> torch.Tensor:
        self.states[key] = state = self.prepare_state_tensor_(
            state=state, init=init, shape=shape
        )
        return state

    def add_to_state(self, target: str, x: torch.Tensor) -> None:
        """Add x to state tensor. Mostly used by Connection objects to implement synaptic transmission."""
        self.states[target] += x

    def scale_and_add_to_state(
        self, scale: float, target: str, x: torch.Tensor
    ) -> None:
        """Add x to state tensor. Mostly used by Connection objects to implement synaptic transmission."""
        self.add_to_state(target, scale * x)

    def clear_input(self) -> None:
        self.input = self.states["input"] = torch.zeros(
            self.int_shape, device=self.device, dtype=self.dtype
        )

    def reset_state(self, batch_size: int = 1) -> None:
        self.int_shape = (batch_size,) + self.shape
        # ? can remove this? And the function that uses it
        # self.flat_seq_shape = (batch_size, self.nb_steps, self.nb_units)
        self.clear_input()
        for key in self.store_state_sequences:
            self.stored_sequences_[key] = []
        self.clk = 0

    def evolve(self) -> None:
        """Advances simulation of group by one timestep and append output to out_seq."""
        self.forward()
        self.set_state_tensor("out", self.out)
        if self.dropout is not None:
            self.out = self.dropout(self.out)
        for key in self.store_state_sequences:
            self.stored_sequences_[key].append(self.states[key])
        self.clk += 1

    def get_state_sequence(self, key: str) -> torch.Tensor:
        seq = self.stored_sequences_[key]
        if key in self.store_state_sequences:
            # if this a list of states, concatenate it along time dimension and store result as tensor for caching
            if type(seq) == list:
                # seq = self.stored_sequences_[key] = torch.stack(seq, dim=1)
                seq = torch.stack(seq, dim=1)
            return seq
        else:
            print(
                "Warning requested state sequence was not stored. Add 'key' to  store_state_sequences list."
            )
            return None

    def get_in_sequence(self) -> Optional[torch.Tensor]:
        return self.get_state_sequence("input")

    def get_out_sequence(self) -> Optional[torch.Tensor]:
        return self.get_state_sequence("out")

    def get_flattened_out_sequence(self) -> Optional[torch.Tensor]:
        return self.get_state_sequence("out").reshape(self.flat_seq_shape)

    def get_firing_rates(self) -> Optional[torch.Tensor]:
        tmp = self.get_out_sequence()
        if tmp is not None:
            rates = torch.mean(tmp, dim=1) / self.time_step  # Average over time
            return rates
        else:
            return None

    def get_mean_population_rate(self) -> Optional[torch.Tensor]:
        rates = self.get_firing_rates()
        if rates is not None:
            return torch.mean(rates)
        else:
            return None

    def get_out_channels(self) -> int:
        return self.shape[0]

    def __call__(self, inputs) -> None:
        raise NotImplementedError
