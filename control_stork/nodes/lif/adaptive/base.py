import torch
from .... import activations
from ..base import CellGroup
from ....extratypes import *

import torch
from typing import Optional, Tuple, Union, Iterable


class AdaptiveLIFGroup(CellGroup):
    def __init__(
        self,
        shape,
        tau_mem=10e-3,
        tau_syn=5e-3,
        tau_ada=100e-3,
        threshold=1.0,
        threshold_decay=0.0,
        threshold_xi=0.0,
        learn_mem: bool = False,
        learn_syn: bool = False,
        learn_ada: bool = False,
        mem_param: str = "single",
        syn_param: str = "single",
        ada_param: str = "single",
        reset: str = "sub",
        diff_reset: bool = False,
        activation=activations.SigmoidSpike,
        dropout_p: float = 0.0,
        stateful: bool = False,
        name: str = "AdaptiveLIFGroup",
        regularizers: Optional[list] = None,
        **kwargs
    ):
        super().__init__(
            shape,
            dropout_p=dropout_p,
            stateful=stateful,
            name=name,
            regularizers=regularizers,
            spiking=True,
            **kwargs
        )

        # Convert initial tau values to log-space
        log_tau_mem_init = torch.log(torch.tensor(tau_mem, dtype=torch.float32))
        log_tau_syn_init = torch.log(torch.tensor(tau_syn, dtype=torch.float32))
        log_tau_ada_init = torch.log(torch.tensor(tau_ada, dtype=torch.float32))

        # Per-layer vs. per-neuron learnability
        self.mem_param = mem_param.lower()
        self.syn_param = syn_param.lower()
        self.ada_param = ada_param.lower()

        assert self.mem_param in ["full", "single"], "mem_param must be 'full' or 'single'"
        assert self.syn_param in ["full", "single"], "syn_param must be 'full' or 'single'"
        assert self.ada_param in ["full", "single"], "ada_param must be 'full' or 'single'"

        # Define shapes for learnable parameters
        mem_shape = self.shape if self.mem_param == "full" else (1,)
        syn_shape = self.shape if self.syn_param == "full" else (1,)
        ada_shape = self.shape if self.ada_param == "full" else (1,)

        # Learnable or fixed tau parameters
        if learn_mem:
            self.log_tau_mem = torch.nn.Parameter(log_tau_mem_init.expand(mem_shape).clone())
        else:
            self.register_buffer("log_tau_mem", log_tau_mem_init.expand(mem_shape).clone())

        if learn_syn:
            self.log_tau_syn = torch.nn.Parameter(log_tau_syn_init.expand(syn_shape).clone())
        else:
            self.register_buffer("log_tau_syn", log_tau_syn_init.expand(syn_shape).clone())

        if learn_ada:
            self.log_tau_ada = torch.nn.Parameter(log_tau_ada_init.expand(ada_shape).clone())
        else:
            self.register_buffer("log_tau_ada", log_tau_ada_init.expand(ada_shape).clone())

        self.threshold = threshold
        self.register_buffer("threshold_decay", torch.tensor(threshold_decay, dtype=torch.float32))
        self.register_buffer("threshold_xi", torch.tensor(threshold_xi, dtype=torch.float32))

        assert self.threshold_decay >= 0.0, "threshold_decay must be non-negative"
        assert self.threshold_xi >= 0.0, "threshold_xi must be non-negative"

        # Reset type
        if reset not in ["sub", "set"]:
            raise ValueError("reset must be either 'sub' or 'set'")
        self.reset_mem = self.subtractive_reset if reset == "sub" else self.multiplicative_reset
        self.diff_reset = diff_reset

        # Spiking activation
        self.activation = activation
        self.spk_nl = self.activation.apply

        # Initialize states
        self.mem = None
        self.syn = None
        self.out = None
        self.rst = None
        self.bt = None
        self.nt = None
        self.vt = None

    def configure(self, time_step: float, device: torch.device, dtype):
        """Configure the module for a given time step and device"""
        self.dt = time_step
        self.device = device
        self.dtype = dtype

        self.to(device)  # Ensure all parameters are on the correct device

    def apply_constraints(self):
        """Clamp tau values to prevent extreme instability"""
        max_log_tau = 3.0  # Ensuring tau remains in a reasonable range
        min_log_tau = -7.0  # Prevent too small tau values

        if isinstance(self.log_tau_mem, torch.nn.Parameter):
            self.log_tau_mem.data = torch.clamp(self.log_tau_mem.data, min=min_log_tau, max=max_log_tau)
        if isinstance(self.log_tau_syn, torch.nn.Parameter):
            self.log_tau_syn.data = torch.clamp(self.log_tau_syn.data, min=min_log_tau, max=max_log_tau)
        if isinstance(self.log_tau_ada, torch.nn.Parameter):
            self.log_tau_ada.data = torch.clamp(self.log_tau_ada.data, min=min_log_tau, max=max_log_tau)

    def tau_to_beta(self, tau: torch.Tensor) -> torch.Tensor:
        """Convert tau to beta for discrete updates"""
        return torch.exp(-self.dt / tau)

    def subtractive_reset(self, mem, rst):
        """Subtractive membrane reset"""
        return mem - self.vt * rst

    def multiplicative_reset(self, mem, rst):
        """Multiplicative membrane reset"""
        return mem * (1.0 - rst)

    def get_spike_and_reset(self, mem: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute spike output and reset signal"""
        mthr = mem - self.vt
        out = self.spk_nl(mthr)
        rst = out if self.diff_reset else out.detach()  # Detach reset for stability
        return out, rst

    def reset_state(self, batch_size: int = 1) -> None:
        """Reset the internal state of the neuron group"""
        super().reset_state(batch_size)
        self.apply_constraints()  # Ensure tau values are in valid range

        self.mem = self.get_state_tensor("mem", state=self.mem)
        self.syn = self.get_state_tensor("syn", state=self.syn)
        self.out = self.rst = self.get_state_tensor("out", state=self.out)

        if self.threshold_decay > 0.0:
            self.bt = self.get_state_tensor("bt", state=self.bt, init=self.threshold)
        else:
            self.bt = self.threshold

        if self.threshold_xi > 0.0:
            self.nt = self.get_state_tensor("nt", state=self.nt)
        else:
            self.nt = 0.0

        self.vt = self.states["vt"] = self.bt + self.threshold_xi * self.nt

    def forward(self) -> None:
        """Forward pass of the adaptive LIF neuron"""
        self.syn = self.states["syn"] = self.syn * self.beta_syn + self.input
        self.mem = self.states["mem"] = (
            self.mem * self.beta_mem + self.syn * (1.0 - self.beta_mem)
        )
        self.out, self.rst = self.get_spike_and_reset(self.mem)
        self.mem = self.states["mem"] = self.reset_mem(self.mem, self.rst)
        self.states["out"] = self.out

        self.bt = self.states["bt"] = self.bt - self.threshold_decay * self.dt + (self.threshold - self.bt) * self.rst
        self.nt = self.states["nt"] = self.nt * self.beta_ada + (1.0 - self.beta_ada) * self.rst
        self.vt = self.states["vt"] = self.bt + self.threshold_xi * self.nt

    @property
    def tau_mem(self):
        return torch.exp(self.log_tau_mem)

    @property
    def tau_syn(self):
        return torch.exp(self.log_tau_syn)

    @property
    def tau_ada(self):
        return torch.exp(self.log_tau_ada)

    @property
    def beta_mem(self):
        return self.tau_to_beta(self.tau_mem)

    @property
    def beta_syn(self):
        return self.tau_to_beta(self.tau_syn)

    @property
    def beta_ada(self):
        return self.tau_to_beta(self.tau_ada)