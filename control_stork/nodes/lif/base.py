import numpy as np
import torch
from torch.nn import Parameter

from ... import activations
from ..base import CellGroup
from ...extratypes import *


class LIFGroup(CellGroup):
    def __init__(
        self,
        shape,
        tau_mem=10e-3,
        tau_syn=5e-3,
        diff_reset=False,
        learn_timescales=False,
        clamp_mem=False,
        activation=activations.SuperSpike,
        dropout_p=0.0,
        stateful=False,
        name="LIFGroup",
        regularizers=None,
        **kwargs
    ):
        """
        Leaky Integrate-and-Fire neuron with decaying synaptic input current.
        It has three state variables that are scalars and are updated at every time step:
        `mem` is for the membrane potential, `syn` is for the synaptic input current, and `out` is 0/1 depending on
        whether the neuron produces a spike.

        Args:
            :param shape: The number of units in this group
            :type shape: int or tuple of int
            :param tau_mem: The membrane time constant in s, defaults to 10e-3
            :type tau_mem: float
            :param tau_syn: The synaptic time constant in s, defaults to 5e-3
            :type tau_syn: float
            :param diff_reset: Whether or not to differentiate through the reset term, defaults to False
            :type diff_reset: bool
            :param learn_timescales: Whether to learn the membrane and synaptic time constants, defaults to False
            :type learn_timescales: bool
            :param activation: The surrogate derivative enabled activation function, defaults to stork.activations.SuperSpike
            :type activation: stork.activations
            :param dropout_p: probability that some elements of the input will be zeroed, defaults to 0.0
            :type dropout_p: float
            :param stateful: Whether or not to reset the state of the neurons between mini-batches, defaults to False
            :type stateful: bool
            :param regularizers: List of regularizers
        """

        super().__init__(
            shape,
            dropout_p=dropout_p,
            stateful=stateful,
            name=name,
            regularizers=regularizers,
            **kwargs
        )
        self.tau_mem = tau_mem
        self.tau_syn = tau_syn
        self.activation = activation
        self.spk_nl = self.activation.apply
        self.diff_reset = diff_reset
        self.learn_timescales = learn_timescales
        self.clamp_mem = clamp_mem
        self.mem = None
        self.syn = None

    def configure(self, time_step, device, dtype):
        self.dcy_mem = float(torch.exp(-time_step / torch.tensor(self.tau_mem)))
        self.scl_mem = 1.0 - self.dcy_mem
        self.dcy_syn = float(torch.exp(-time_step / torch.tensor(self.tau_syn)))
        self.scl_syn = 1.0 - self.dcy_syn
        if self.learn_timescales:
            mem_param = torch.randn(1, device=device, dtype=dtype, requires_grad=True)
            syn_param = torch.randn(1, device=device, dtype=dtype, requires_grad=True)
            self.mem_param = Parameter(mem_param, requires_grad=self.learn_timescales)
            self.syn_param = Parameter(syn_param, requires_grad=self.learn_timescales)
        super().configure(time_step, device, dtype)

    def reset_state(self, batch_size: int = 1):
        super().reset_state(batch_size)
        if self.learn_timescales:
            self.dcy_mem = torch.exp(
                -self.time_step / (2 * self.tau_mem * torch.sigmoid(self.mem_param))
            )
            self.scl_mem = 1.0 - self.dcy_mem
            self.dcy_syn = torch.exp(
                -self.time_step / (2 * self.tau_syn * torch.sigmoid(self.syn_param))
            )
            self.scl_syn = 1.0 - self.dcy_syn
        self.mem = self.get_state_tensor("mem", state=self.mem)
        self.syn = self.get_state_tensor("syn", state=self.syn)
        self.out = self.states["out"] = torch.zeros(
            self.int_shape, device=self.device, dtype=self.dtype
        )

    def get_spike_and_reset(self, mem):
        mthr = mem - 1.0

        out = self.spk_nl(mthr)
        if self.diff_reset:
            rst = out
        else:
            # if differentiation should not go through reset term, detach it from the computational graph
            rst = out.detach()

        return out, rst

    def forward(self):
        # spike & reset
        new_out, rst = self.get_spike_and_reset(self.mem)

        # synaptic & membrane dynamics
        new_syn = self.dcy_syn * self.syn + self.input
        new_mem = (self.dcy_mem * self.mem + self.scl_mem * self.syn) * (
            1.0 - rst
        )  # multiplicative reset

        # Clamp membrane potential
        if self.clamp_mem:
            new_mem = torch.clamp(new_mem, max=1.01)

        self.out = self.states["out"] = new_out
        self.mem = self.states["mem"] = new_mem
        self.syn = self.states["syn"] = new_syn


class FastLIFGroup(LIFGroup):
    def __init__(
        self,
        shape,
        tau_mem=10e-3,
        tau_syn=5e-3,
        diff_reset=False,
        learn_timescales=False,
        clamp_mem=False,
        activation=activations.SuperSpike,
        dropout_p=0.0,
        stateful=False,
        name="LIFGroup",
        regularizers=None,
        **kwargs
    ):
        """
        Leaky Integrate-and-Fire neuron with decaying synaptic input current.
        It has three state variables that are scalars and are updated at every time step:
        `mem` is for the membrane potential, `syn` is for the synaptic input current, and `out` is 0/1 depending on
        whether the neuron produces a spike.

        Args:
            :param shape: The number of units in this group
            :type shape: int or tuple of int
            :param tau_mem: The membrane time constant in s, defaults to 10e-3
            :type tau_mem: float
            :param tau_syn: The synaptic time constant in s, defaults to 5e-3
            :type tau_syn: float
            :param diff_reset: Whether or not to differentiate through the reset term, defaults to False
            :type diff_reset: bool
            :param learn_timescales: Whether to learn the membrane and synaptic time constants, defaults to False
            :type learn_timescales: bool
            :param activation: The surrogate derivative enabled activation function, defaults to stork.activations.SuperSpike
            :type activation: stork.activations
            :param dropout_p: probability that some elements of the input will be zeroed, defaults to 0.0
            :type dropout_p: float
            :param stateful: Whether or not to reset the state of the neurons between mini-batches, defaults to False
            :type stateful: bool
            :param regularizers: List of regularizers
        """
        super().__init__(
            shape,
            tau_mem,
            tau_syn,
            diff_reset,
            learn_timescales,
            clamp_mem,
            activation,
            dropout_p,
            stateful,
            name,
            regularizers,
            **kwargs
        )

    def forward(self):

        # synaptic & membrane dynamics
        new_syn = self.dcy_syn * self.syn + self.input
        new_mem = self.dcy_mem * self.mem + self.scl_mem * new_syn

        # spike & reset
        new_out, rst = self.get_spike_and_reset(new_mem)

        new_mem = new_mem * (1.0 - rst)  # multiplicative reset

        # Clamp membrane potential
        if self.clamp_mem:
            new_mem = torch.clamp(new_mem, max=1.01)

        self.out = self.states["out"] = new_out
        self.mem = self.states["mem"] = new_mem
        self.syn = self.states["syn"] = new_syn


class NoisyFastLIFGroup(FastLIFGroup):
    def __init__(
        self,
        shape,
        tau_mem=10e-3,
        tau_syn=5e-3,
        diff_reset=False,
        learn_timescales=False,
        clamp_mem=False,
        activation=activations.SuperSpike,
        dropout_p=0.0,
        stateful=False,
        name="LIFGroup",
        regularizers=None,
        noise_std: float = 0.1,
        **kwargs
    ):
        """
        Leaky Integrate-and-Fire neuron with decaying synaptic input current.
        It has three state variables that are scalars and are updated at every time step:
        `mem` is for the membrane potential, `syn` is for the synaptic input current, and `out` is 0/1 depending on
        whether the neuron produces a spike.

        Args:
            :param shape: The number of units in this group
            :type shape: int or tuple of int
            :param tau_mem: The membrane time constant in s, defaults to 10e-3
            :type tau_mem: float
            :param tau_syn: The synaptic time constant in s, defaults to 5e-3
            :type tau_syn: float
            :param diff_reset: Whether or not to differentiate through the reset term, defaults to False
            :type diff_reset: bool
            :param learn_timescales: Whether to learn the membrane and synaptic time constants, defaults to False
            :type learn_timescales: bool
            :param activation: The surrogate derivative enabled activation function, defaults to stork.activations.SuperSpike
            :type activation: stork.activations
            :param dropout_p: probability that some elements of the input will be zeroed, defaults to 0.0
            :type dropout_p: float
            :param stateful: Whether or not to reset the state of the neurons between mini-batches, defaults to False
            :type stateful: bool
            :param regularizers: List of regularizers
            :param noise_std: Standard deviation of the noise added to the threshold
            :type noise_std: float
        """
        super().__init__(
            shape,
            tau_mem,
            tau_syn,
            diff_reset,
            learn_timescales,
            clamp_mem,
            activation,
            dropout_p,
            stateful,
            name,
            regularizers,
            **kwargs
        )

        self.noise_std = noise_std

    def get_spike_and_reset(self, mem):
        threshold = 1.0 + self.noise_std * torch.randn_like(mem)
        mthr = mem - threshold
        out = self.spk_nl(mthr)

        if self.diff_reset:
            rst = out
        else:
            # if differentiation should not go through reset term, detach it from the computational graph
            rst = out.detach()

        return out, rst
