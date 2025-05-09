import numpy as np
import torch

from . import CellGroup
from ..extratypes import *


class BaseReadoutGroup(CellGroup):
    """
    Base class for readout groups handling weight and output scaling.

    Initializes and manages learnable or fixed scaling parameters.
    """

    def __init__(
        self,
        shape: Union[int, Iterable[int]],
        weight_scale: float = 1.0,
        output_scale: float = 1.0,
        apply_tanh: bool = False,
        learn_weight_scale: bool = False,
        learn_output_scale: bool = False,
        stateful: bool = False,
        name: Optional[str] = None,
        store_sequences: Optional[Iterable[str]] = ("out",),
        **kwargs: Any,
    ) -> None:
        """
        Args:
            shape:               Number or shape iterable for output units.
            weight_scale:        Initial weight scaling factor.
            output_scale:        Initial output scaling factor (used if apply_tanh=True).
            apply_tanh:          Whether to apply tanh activation on output.
            learn_weight_scale:  If True, weight_scale is learnable.
            learn_output_scale:  If True, output_scale is learnable.
            stateful:            Whether to maintain state across calls.
            name:                Optional name for the group.
            store_sequences:     Iterable of state names to record sequences.
            **kwargs:            Additional CellGroup kwargs.
        """
        super().__init__(
            shape,
            stateful=stateful,
            name=name or self.__class__.__name__,
            store_sequences=store_sequences,
            **kwargs,
        )
        self.apply_tanh = apply_tanh
        # placeholders
        self.out: Optional[Tensor] = None
        self.mem: Optional[Tensor] = None
        # initialize scaling parameters
        self.setup_scaling(
            weight_scale,
            output_scale,
            learn_weight_scale,
            learn_output_scale,
        )

    def setup_scaling(
        self, weight_scale: float, output_scale: float, learn_ws: bool, learn_os: bool
    ) -> None:
        """
        Initialize or update the weight/output scaling parameters.

        Args:
            weight_scale (float): New weight scaling factor.
            output_scale (float): New output scaling factor.
            learn_ws (bool): Whether weight_scale is learnable.
            learn_os (bool): Whether output_scale is learnable.
        """
        ws = torch.log(torch.full((self.nb_units,), weight_scale, dtype=torch.float32))
        os = torch.log(torch.full((self.nb_units,), output_scale, dtype=torch.float32))
        # weight scale
        if hasattr(self, "log_weight_scale") and isinstance(
            self.log_weight_scale, torch.nn.Parameter
        ):
            with torch.no_grad():
                self.log_weight_scale.copy_(ws)
        else:
            if hasattr(self, "log_weight_scale"):
                del self.log_weight_scale
            if learn_ws:
                self.log_weight_scale = torch.nn.Parameter(ws.clone())
            else:
                self.register_buffer("log_weight_scale", ws.clone())
        # output scale
        if not self.apply_tanh:
            if not hasattr(self, "log_output_scale"):
                self.register_buffer("log_output_scale", os.clone())
            return
        if hasattr(self, "log_output_scale") and isinstance(
            self.log_output_scale, torch.nn.Parameter
        ):
            with torch.no_grad():
                self.log_output_scale.copy_(os)
        else:
            if hasattr(self, "log_output_scale"):
                del self.log_output_scale
            if learn_os:
                self.log_output_scale = torch.nn.Parameter(os.clone())
            else:
                self.register_buffer("log_output_scale", os.clone())

    @property
    def weight_scale(self) -> Tensor:
        """Current weight scaling tensor."""
        return torch.exp(self.log_weight_scale)

    @property
    def output_scale(self) -> Tensor:
        """Current output scaling tensor."""
        return torch.exp(self.log_output_scale)

    def reset_state(self, batch_size: int = 1) -> None:
        """Reset state tensors; subclasses may override."""
        super().reset_state(batch_size)

    def forward(self) -> None:
        """Compute output; to be implemented by subclasses."""
        raise NotImplementedError


class ReadoutGroup(BaseReadoutGroup):
    """
    Readout group with shared ('single') or per-unit ('full')
    membrane and synaptic time constants, plus scaling.
    """

    def __init__(
        self,
        shape: Union[int, Iterable[int]],
        tau_mem: Union[float, Tensor] = 1e-2,
        tau_syn: Union[float, Tensor] = 5e-3,
        mem_param: str = "single",
        syn_param: str = "single",
        weight_scale: float = 1.0,
        output_scale: float = 1.0,
        apply_tanh: bool = False,
        learn_mem: bool = False,
        learn_syn: bool = False,
        learn_weight_scale: bool = False,
        learn_output_scale: bool = False,
        stateful: bool = False,
        name: Optional[str] = None,
        store_sequences: Optional[Iterable[str]] = ("out",),
        **kwargs: Any,
    ) -> None:
        super().__init__(
            shape,
            weight_scale=weight_scale,
            output_scale=output_scale,
            apply_tanh=apply_tanh,
            learn_weight_scale=learn_weight_scale,
            learn_output_scale=learn_output_scale,
            stateful=stateful,
            name=name or "Readout",
            store_sequences=store_sequences,
            **kwargs,
        )
        # parameter modes
        self.mem_param = mem_param.lower()
        self.syn_param = syn_param.lower()
        assert self.mem_param in (
            "single",
            "full",
        ), "mem_param must be 'single' or 'full'"
        assert self.syn_param in (
            "single",
            "full",
        ), "syn_param must be 'single' or 'full'"
        # initialize tau parameters
        self._init_tau("mem", tau_mem, learn_mem, self.mem_param)
        self._init_tau("syn", tau_syn, learn_syn, self.syn_param)
        # placeholders
        self.syn: Optional[Tensor] = None
        self.mem: Optional[Tensor] = None
        self.out: Optional[Tensor] = None

    def _init_tau(
        self, name: str, init_val: Union[float, Tensor], learn: bool, mode: str
    ) -> None:
        """
        Initialize a tau parameter ('mem' or 'syn').

        Args:
            name:     'mem' or 'syn'.
            init_val: scalar float or Tensor of size 1 or nb_units.
            learn:    if True, make it a Parameter; else register as buffer.
            mode:     'single' or 'full'.
        """
        # log-domain tensor
        if isinstance(init_val, Tensor):
            num = init_val.numel()
            if mode == "full":
                assert num in (
                    1,
                    self.nb_units,
                ), f"init_val for '{name}' must have 1 or {self.nb_units} elements"
            else:
                assert num == 1, f"init_val for '{name}' must be scalar in single mode"
            log_init = torch.log(init_val.float())
        else:
            log_init = torch.log(torch.tensor(init_val, dtype=torch.float32))
        # shape
        if mode == "full":
            if log_init.numel() == 1:
                log_tensor = log_init.flatten()[0].expand((self.nb_units,)).clone()
            else:
                log_tensor = log_init.clone()
        else:
            val = log_init.flatten()[0]
            log_tensor = val.expand((1,)).clone()
        # register
        pname = f"log_tau_{name}"
        if learn:
            setattr(self, pname, torch.nn.Parameter(log_tensor))
        else:
            self.register_buffer(pname, log_tensor)

    def configure(self, time_step: float, device: torch.device, dtype: Any) -> None:
        super().configure(time_step, device, dtype)
        self.apply_constraints()
        self.update_tau_and_beta()
        self.to(device)

    def apply_constraints(self) -> None:
        """Clamp log-tau values to reasonable bounds."""
        with torch.no_grad():
            self.log_tau_mem.clamp_(-7.0, 2.0)
            self.log_tau_syn.clamp_(-7.0, 2.0)

    def update_tau_and_beta(self) -> None:
        """Recompute tau, beta, and scale from log-tau."""
        self.tau_mem = torch.exp(self.log_tau_mem)
        self.tau_syn = torch.exp(self.log_tau_syn)
        self.beta_mem = torch.exp(-self.dt / self.tau_mem)
        self.beta_syn = torch.exp(-self.dt / self.tau_syn)
        self.scl_mem = 1.0 - self.beta_mem

    def reset_state(self, batch_size: int = 1) -> None:
        super().reset_state(batch_size)
        self.apply_constraints()
        self.update_tau_and_beta()
        self.syn = self.get_state_tensor("syn", self.syn)
        self.mem = self.get_state_tensor("mem", self.mem)
        self.out = self.get_state_tensor("out", self.out)

    def forward(self) -> None:
        new_syn = self.beta_syn * self.syn + self.input * self.weight_scale
        new_mem = self.beta_mem * self.out + self.scl_mem * new_syn
        self.syn = self.states["syn"] = new_syn
        self.mem = self.states["mem"] = new_mem
        if self.apply_tanh:
            self.out = self.states["out"] = torch.tanh(new_mem) * self.output_scale
        else:
            self.out = self.states["out"] = new_mem


class DirectReadoutGroup(BaseReadoutGroup):
    """
    Direct linear readout with optional tanh activation.

    Inherits scaling logic from BaseReadoutGroup.
    """

    def __init__(
        self,
        shape: Union[int, Iterable[int]],
        weight_scale: float = 1.0,
        output_scale: float = 1.0,
        apply_tanh: bool = False,
        learn_weight_scale: bool = False,
        learn_output_scale: bool = False,
        initial_state: float = -1e-3,
        stateful: bool = False,
        name: Optional[str] = None,
        store_sequences: Optional[Iterable[str]] = ("out",),
        **kwargs: Any,
    ) -> None:
        super().__init__(
            shape,
            weight_scale=weight_scale,
            output_scale=output_scale,
            apply_tanh=apply_tanh,
            learn_weight_scale=learn_weight_scale,
            learn_output_scale=learn_output_scale,
            stateful=stateful,
            name=name or "DirectReadout",
            store_sequences=store_sequences,
            **kwargs,
        )
        self.initial_state = initial_state

    def reset_state(self, batch_size: int = 1) -> None:
        super().reset_state(batch_size)
        self.out = self.get_state_tensor("out", self.out, init=self.initial_state)

    def forward(self) -> None:
        mem = self.input * self.weight_scale
        if self.apply_tanh:
            self.out = self.states["out"] = torch.tanh(mem) * self.output_scale
        else:
            self.out = self.states["out"] = mem
        self.mem = self.states["mem"] = mem


class TimeAverageReadoutGroup(BaseReadoutGroup):
    """
    Time-average readout: maintains a buffer of past inputs and outputs their average.

    Inherits scaling logic from BaseReadoutGroup.
    """

    def __init__(
        self,
        shape: Union[int, Iterable[int]],
        weight_scale: float = 1.0,
        output_scale: float = 1.0,
        apply_tanh: bool = False,
        learn_weight_scale: bool = False,
        learn_output_scale: bool = False,
        steps: int = 1,
        initial_state: float = -1e-3,
        stateful: bool = False,
        name: Optional[str] = None,
        store_sequences: Optional[Iterable[str]] = ("out",),
        **kwargs: Any,
    ) -> None:
        super().__init__(
            shape,
            weight_scale=weight_scale,
            output_scale=output_scale,
            apply_tanh=apply_tanh,
            learn_weight_scale=learn_weight_scale,
            learn_output_scale=learn_output_scale,
            stateful=stateful,
            name=name or "TimeAverageReadout",
            store_sequences=store_sequences,
            **kwargs,
        )
        self.steps = steps
        self.initial_state = initial_state
        self.memory: list[Tensor] = []

    def reset_state(self, batch_size: int = 1) -> None:
        super().reset_state(batch_size)
        start = self.get_state_tensor("out", self.out, init=self.initial_state)
        self.memory = [start] * self.steps
        self.out = start

    def forward(self) -> None:
        mem = self.input * self.weight_scale
        self.memory.pop(0)
        self.memory.append(mem)
        avg_mem = torch.mean(torch.stack(self.memory), dim=0)
        if self.apply_tanh:
            self.out = self.states["out"] = torch.tanh(avg_mem) * self.output_scale
        else:
            self.out = self.states["out"] = avg_mem
        self.mem = self.states["mem"] = avg_mem
