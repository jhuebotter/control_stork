import torch
from .... import activations
from ..base import CellGroup
from ....extratypes import *

import torch
from typing import Optional, Tuple, Union, Iterable


class AdaptiveLIFGroup(CellGroup):
    """
    Adaptive Leaky Integrate-and-Fire group with configurable time-constant parameters.

    Tau parameters (tau_mem, tau_syn, tau_ada) can be shared per-group ('single')
    or individualized per-neuron ('full'), and can be learnable or fixed.
    """

    def __init__(
        self,
        shape: Union[int, Iterable[int]],
        tau_mem: Union[float, Tensor] = 10e-3,
        tau_syn: Union[float, Tensor] = 5e-3,
        tau_ada: Union[float, Tensor] = 100e-3,
        threshold: float = 1.0,
        threshold_decay: float = 0.0,
        threshold_xi: float = 0.0,
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
        **kwargs,
    ):
        super().__init__(
            shape,
            dropout_p=dropout_p,
            stateful=stateful,
            name=name,
            regularizers=regularizers,
            spiking=True,
            **kwargs,
        )
        # Store parameterization modes
        self.mem_param = mem_param.lower()
        self.syn_param = syn_param.lower()
        self.ada_param = ada_param.lower()
        for mode, label in [
            (self.mem_param, "mem_param"),
            (self.syn_param, "syn_param"),
            (self.ada_param, "ada_param"),
        ]:
            assert mode in ("single", "full"), f"{label} must be 'single' or 'full'"

        # Initialize tau parameters via helper
        self._init_tau("mem", tau_mem, learn_mem, self.mem_param)
        self._init_tau("syn", tau_syn, learn_syn, self.syn_param)
        self._init_tau("ada", tau_ada, learn_ada, self.ada_param)

        # Other parameters
        self.dt = None
        self.device = None
        self.dtype = None
        self.threshold = threshold
        self.register_buffer(
            "threshold_decay", torch.tensor(threshold_decay, dtype=torch.float32)
        )
        self.register_buffer(
            "threshold_xi", torch.tensor(threshold_xi, dtype=torch.float32)
        )
        self.reset_mem = (
            self.subtractive_reset if reset == "sub" else self.multiplicative_reset
        )
        self.diff_reset = diff_reset
        self.activation = activation
        self.spk_nl = activation.apply

        # Placeholders for state variables
        self.mem = self.syn = self.out = self.rst = None
        self.bt = self.nt = self.vt = None

    def _init_tau(
        self, name: str, init_val: Union[float, Tensor], learn: bool, mode: str
    ) -> None:
        """
        Initialize a tau parameter (mem, syn, or ada) from a float or Tensor.

        Args:
            name:     'mem', 'syn', or 'ada'.
            init_val: initial tau value; scalar float or Tensor of shape matching group units.
            learn:    if True, make it an nn.Parameter; else register as buffer.
            mode:     'single' for shared, 'full' for per-neuron values.
        """
        # prepare log-domain tensor
        if isinstance(init_val, Tensor):
            num = init_val.numel()
            if mode == "full":
                assert (
                    num == 1 or num == self.nb_units
                ), f"init_val tensor for '{name}' must have 1 or {self.nb_units} elements in full mode, got {num}"
            else:
                assert (
                    num == 1
                ), f"init_val tensor for '{name}' must be scalar (1 element) in single mode, got {num}"
            log_init = torch.log(init_val.to(dtype=torch.float32))
        else:
            log_init = torch.log(torch.tensor(init_val, dtype=torch.float32))

        # determine target shape
        if mode == "full":
            if log_init.dim() == 0:
                size = (self.nb_units,)
                log_tensor = log_init.expand(size).clone()
            else:
                if log_init.numel() == 1:
                    size = (self.nb_units,)
                    log_tensor = log_init.flatten()[0].expand(size).clone()
                else:
                    log_tensor = log_init.clone()
        else:
            val = log_init.flatten()[0]
            log_tensor = val.expand((1,)).clone()

        # register as parameter or buffer
        param_name = f"log_tau_{name}"
        if learn:
            setattr(self, param_name, torch.nn.Parameter(log_tensor))
        else:
            self.register_buffer(param_name, log_tensor)

    def configure(self, time_step: float, device: torch.device, dtype):
        super().configure(time_step, device, dtype)
        self.apply_constraints()
        self.update_tau_and_beta()
        self.to(device)

    def apply_constraints(self):
        min_log, max_log = -7.0, 2.0
        with torch.no_grad():
            for nm in ("mem", "syn", "ada"):
                param = getattr(self, f"log_tau_{nm}")
                param.clamp_(min=min_log, max=max_log)

    def update_tau_and_beta(self):
        self.tau_mem = torch.exp(self.log_tau_mem)
        self.tau_syn = torch.exp(self.log_tau_syn)
        self.tau_ada = torch.exp(self.log_tau_ada)
        self.beta_mem = self.tau_to_beta(self.tau_mem)
        self.beta_syn = self.tau_to_beta(self.tau_syn)
        self.beta_ada = self.tau_to_beta(self.tau_ada)

    def tau_to_beta(self, tau: Tensor) -> Tensor:
        return torch.exp(-self.dt / tau)

    def subtractive_reset(self, mem, rst):
        return mem - self.vt * rst

    def multiplicative_reset(self, mem, rst):
        return mem * (1.0 - rst)

    def get_spike_and_reset(self, mem: Tensor) -> Tuple[Tensor, Tensor]:
        mthr = mem - self.vt
        out = self.spk_nl(mthr)
        rst = out if self.diff_reset else out.detach()
        return out, rst

    def reset_state(self, batch_size: int = 1) -> None:
        super().reset_state(batch_size)
        self.apply_constraints()
        self.update_tau_and_beta()
        self.mem = self.get_state_tensor("mem", self.mem)
        self.syn = self.get_state_tensor("syn", self.syn)
        self.out, self.rst = self.get_state_tensor("out", self.out), None
        self.bt = (
            self.get_state_tensor("bt", self.bt, init=self.threshold)
            if self.threshold_decay > 0
            else self.threshold
        )
        self.nt = self.get_state_tensor("nt", self.nt) if self.threshold_xi > 0 else 0.0
        self.vt = self.states["vt"] = self.bt + self.threshold_xi * self.nt

    def forward(self) -> None:
        self.syn = self.states["syn"] = self.syn * self.beta_syn + self.input
        self.mem = self.states["mem"] = self.mem * self.beta_mem + self.syn * (
            1.0 - self.beta_mem
        )
        self.out, self.rst = self.get_spike_and_reset(self.mem)
        self.mem = self.states["mem"] = self.reset_mem(self.mem, self.rst)
        self.states["out"] = self.out
        self.bt = self.states["bt"] = (
            self.bt
            - self.threshold_decay * self.dt
            + (self.threshold - self.bt) * self.rst
        )
        self.nt = self.states["nt"] = (
            # self.nt * self.beta_ada + (1.0 - self.beta_ada) * self.rst
            self.nt * self.beta_ada
            + self.out
        )
        self.vt = self.states["vt"] = self.bt + self.threshold_xi * self.nt
