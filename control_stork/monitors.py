import torch
import torch.nn as nn

from .extratypes import *
from .nodes import CellGroup

# TODO: add docstrings


class Monitor:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        raise NotImplementedError

    def execute(self) -> None:
        raise NotImplementedError

    def get_data(self) -> None:
        raise NotImplementedError


class SpikeMonitor(Monitor):
    """Records spikes in sparse RAS format

    Args:
        group: The group to record from

    Returns:
        argwhere of out sequence
    """

    def __init__(self, group: CellGroup) -> None:
        super().__init__()
        self.group = group
        self.batch_count_ = 0

    def reset(self) -> None:
        self.data = []

    def execute(self) -> None:
        self.data.append(self.group.states["out"].detach())

    def get_data(self) -> torch.Tensor:
        if not self.data:
            return None
        out = torch.stack(self.data, dim=1).cpu()
        tmp = torch.nonzero(out)
        tmp[:, 0] += self.batch_count_
        self.batch_count_ += out.shape[0]
        return tmp


class StateMonitor(Monitor):
    """Records the state of a neuron group over time

    Args:
        group: The group to record from
        key: The name of the state
    """

    def __init__(
        self, group: CellGroup, key: str, subset: Optional[Union[int, Iterable]] = None
    ) -> None:
        super().__init__()
        self.group = group
        self.key = key
        self.subset = subset

    def reset(self) -> None:
        self.data = []

    def execute(self) -> None:
        if self.subset is not None:
            self.data.append(self.group.states[self.key][:, self.subset].detach())
        else:
            self.data.append(self.group.states[self.key].detach())

    def get_data(self) -> torch.Tensor:
        if not self.data:
            return None
        return torch.stack(self.data, dim=1).cpu()


class PlotStateMonitor(StateMonitor):
    """Records the state of a neuron group over time and plots it

    Args:
        group: The group to record from
        key: The name of the state
    """

    def __init__(
        self,
        group: CellGroup,
        key: str,
        subset: Optional[Union[int, Iterable]] = None,
        plot_fn: Optional[Callable] = None,
        **kwargs,
    ) -> None:
        super().__init__(group, key, subset)
        self.plot_fn = plot_fn
        self.kwargs = kwargs

    def get_data(self) -> "matplotlib.figure.Figure":
        data = super().get_data()
        if data is None:
            return None
        return self.plot_fn(data, **self.kwargs)


class SpikeCountMonitor(Monitor):
    """Counts number of spikes (sum over time in get_out_sequence() for each neuron)

    Args:
        group: The group to record from

    Returns:
        A tensor with spike counts for each input and neuron
    """

    def __init__(self, group: CellGroup) -> None:
        super().__init__()
        self.group = group

    def reset(self) -> None:
        self.data = None

    def execute(self) -> None:
        spk = self.group.states["out"].detach().clone()
        if self.data is None:
            self.data = spk
        else:
            self.data += spk

    def get_data(self) -> torch.Tensor:
        if self.data is None:
            return None
        return self.data.cpu()


class ActiveNeuronMonitor(Monitor):
    """Counts number of active neurons (more than n_min spikes) / number of all neurons

    Args:
        group: The group to record from
        n_min: The minimum number of spikes to be considered active

    Returns:
        A float tensor with the fraction of active neurons for each input
    """

    def __init__(self, group: CellGroup, n_min: int = 1) -> None:
        super().__init__()
        self.group = group
        self.n_min = n_min

    def reset(self) -> None:
        self.n = 0
        self.data = None

    def execute(self) -> None:
        self.n += 1
        spk = self.group.states["out"].detach().clone()
        if self.data is None:
            self.data = spk
        else:
            self.data += spk

    def get_data(self) -> torch.Tensor:
        if self.data is None:
            return None
        all = torch.sum(self.data, dim=0)
        active = (all >= self.n_min).float()
        ratio = torch.sum(active) / self.group.nb_units
        return ratio.cpu()


class PopulationSpikeCountMonitor(Monitor):
    """Counts total number of spikes (sum over time in get_out_sequence() for the group)

    Args:
        group: The group to record from
        per_example: If True, returns the mean spike count per example
        avg: If True, returns the mean spike count over time (spike density)

    Returns:
        A tensor with spike counts for each input and neuron
    """

    def __init__(
        self, group: CellGroup, per_example: bool = False, avg: bool = False
    ) -> None:
        super().__init__()
        self.group = group
        self.per_example = per_example
        self.avg = avg

    def reset(self) -> None:
        self.data = []

    def execute(self) -> None:
        self.data.append(self.group.states["out"].detach())

    def get_data(self) -> torch.Tensor:
        if not self.data:
            return None
        if self.avg:
            s1 = torch.mean(torch.stack(self.data, dim=1), dim=1).cpu()
        else:
            s1 = torch.sum(torch.stack(self.data, dim=1), dim=1).cpu()
        return torch.mean(s1, dim=1) if self.per_example else torch.mean(s1)


class PopulationFiringRateMonitor(Monitor):
    """Monitors population firing rate (nr of spikes / nr of neurons for every time_step)

    Args:
        group: The group to record from

    Returns:
        A tensor with population firing rate for each input and time_step
    """

    def __init__(self, group: CellGroup) -> None:
        super().__init__()
        self.group = group

    def reset(self) -> None:
        self.data = []

    def execute(self) -> None:
        self.data.append(self.group.states["out"].detach())

    def get_data(self) -> torch.Tensor:
        if not self.data:
            return None
        s1 = torch.stack(self.data, dim=1).cpu()
        s1 = s1.reshape(s1.shape[0], s1.shape[1], self.group.nb_units)
        return torch.sum(s1, dim=-1) / self.group.nb_units


class MeanVarianceMonitor(Monitor):
    """Measures mean and variance of input

    Args:
        group: The group to record from
        key (string): State variable to monitor (Monitors mean and variance of a state variable)


    Returns:
        A tensors with mean and variance for each neuron/state along the last dim
    """

    def __init__(self, group: CellGroup, state: str = "input", dim=None) -> None:
        super().__init__()
        self.group = group
        self.key = state
        self.dim = dim

    def reset(self) -> None:
        self.s = 0
        self.s2 = 0
        self.c = 0

    def execute(self) -> None:
        tns = self.group.states[self.key]
        if self.dim is not None:
            tns = torch.mean(tns, dim=self.dim)
        self.s += tns.detach()
        self.s2 += torch.square(tns).detach()
        self.c += 1

    def get_data(self) -> torch.Tensor:
        mean = self.s / self.c
        var = self.s2 / self.c - mean**2
        return torch.stack((mean, var), len(mean.shape)).cpu()


class PropertyMonitor(Monitor):
    """Records a property of a group over time

    Args:
        group: The group to record from
        key: The name of the property
        dtype: The type of the property
    """

    def __init__(self, group: CellGroup, key: str, dtype=None) -> None:
        super().__init__()
        self.group = group
        self.key = key
        self.dtype = dtype
        self.data = None

    def reset(self) -> None:
        self.data = None

    def execute(self) -> None:
        att = getattr(self.group, self.key)
        if torch.is_tensor(att):
            att = att.detach().cpu()
        if self.dtype is not None:
            att = self.dtype(att)
        self.data = att

    def get_data(self) -> torch.Tensor:
        return self.data


class PlotPropertyMonitor(PropertyMonitor):
    """Plots a snapshot property (1D tensor) via a histogram."""

    def __init__(
        self,
        group: CellGroup,
        key: str,
        plot_fn: Callable,
        **kwargs,
    ) -> None:
        super().__init__(group, key)
        self.plot_fn = plot_fn
        self.kwargs = kwargs

    def get_data(self) -> "matplotlib.figure.Figure":
        data = super().get_data()
        if data is None:
            return None
        return self.plot_fn(data, **self.kwargs)


class GradientMonitor(Monitor):
    """Records the gradients (weight.grad)

    Args:
        target: The tensor or nn.Module to record from
                (usually a stork.connection.op object)
                Needs to have a .weight argument
    """

    def __init__(self, target: nn.Module) -> None:
        super().__init__()
        self.target = target

    def reset(self) -> None:
        pass

    def set_hook(self) -> None:
        """
        Sets the backward hook
        """
        pass

    def remove_hook(self) -> None:
        pass

    def execute(self) -> None:
        pass

    def get_data(self) -> torch.Tensor:
        # unsqueeze so that the output from the monitor is [batch_nr x weightmatrix-dims]
        return self.target.weight.grad.detach().cpu().abs().unsqueeze(0)


class GradientOutputMonitor(GradientMonitor):
    """Records the gradients wrt the neuronal output
        computed in the backward pass

    Args:
        target: The tensor or nn.Module to record from
                (usually a stork.connection.op object)
    """

    def __init__(self, target: nn.Module) -> None:
        super().__init__(target)
        self.count = 0
        self.sum = 0

    def set_hook(self) -> None:
        """
        Sets the backward hook
        """
        self.hook = self.target.register_full_backward_hook(self.grab_gradient)

    def remove_hook(self) -> None:
        self.hook.remove()

    def grab_gradient(self, module, grad_input, grad_output) -> None:
        ## TODO: check what the extra arguments are
        mean_grad = grad_output[0].detach().cpu().abs()
        self.sum += mean_grad
        self.count += 1

    def reset(self) -> None:
        self.count = 0
        self.sum = torch.zeros(1)

    def execute(self) -> None:
        pass

    def get_data(self) -> torch.Tensor:
        return self.sum / self.count


class SnapshotStatsMonitor(Monitor):
    """
    One-shot statistics over neuron/channel dimensions at execute() time.

    Computes mean, variance, min, max, and quantiles of a tensor in one call
    and stores them for get_data().

    Args:
        group:       Object (e.g., CellGroup) with attribute or states-key.
        key:         Name of the attribute or state to sample.
        neuron_dims: Dimensions to collapse for neuron-wise stats (default: none).
        quantiles:   Tuple of quantile fractions (default: (0.25, 0.75)).
    """

    def __init__(
        self,
        group: Any,
        key: str,
        neuron_dims: Optional[Sequence[int]] = None,
        quantiles: Tuple[float, float] = (0.25, 0.5, 0.75),
    ) -> None:
        super().__init__()
        self.group = group
        self.key = key
        self.neuron_dims = tuple(neuron_dims) if neuron_dims is not None else (-1,)
        self.neuron_dims = tuple(neuron_dims) if neuron_dims is not None else ()
        self.quantiles = quantiles
        # build dynamic field names based on chosen quantiles
        self.field_names = ["mean", "var", "min", "max"] + [
            f"q{int(q*100)}" for q in self.quantiles
        ]
        self.latest: Dict[str, Tensor] = {}

    def reset(self) -> None:
        """No-op: snapshot uses only latest execute()"""
        self.latest = {}

    def execute(self) -> None:
        """Fetch tensor, optionally collapse neuron dims, compute and store stats."""
        # fetch attribute or state
        x = getattr(self.group, self.key, None)
        if x is None:
            x = self.group.states[self.key]
        # ensure tensor
        if not torch.is_tensor(x):
            t = torch.tensor(x)
        else:
            t = x.detach()
        # collapse specified neuron_dims
        for d in sorted(self.neuron_dims, reverse=True):
            if t.dim() > d:
                t = t.mean(dim=d)
        t = t.cpu()
        # compute stats
        self.latest.clear()
        self.latest["mean"] = t.mean()
        self.latest["var"] = t.var(unbiased=False)
        self.latest["min"] = t.min()
        self.latest["max"] = t.max()
        qs = t.quantile(torch.tensor(self.quantiles))
        for qval, name in zip(qs.tolist(), (f"q{int(q*100)}" for q in self.quantiles)):
            self.latest[name] = torch.tensor(qval)

    def get_data(self) -> Dict[str, Tensor]:
        """Return the latest computed statistics."""
        if not self.latest:
            raise RuntimeError("No data: call execute() before get_data().")
        return self.latest


class StreamingStatsMonitor(Monitor):
    """
    Streaming statistics over multiple execute() calls.

    Accumulates sum, sum-of-squares, min, max, and buffers samples
    to compute quantiles at get_data().

    Args:
        group:       Object with attribute or states-key.
        key:         Name of the attribute or state to monitor.
        reduce_dims: Dimensions to collapse each sample (e.g. batch/time).
        quantiles:   Tuple of quantile fractions (default: (0.25, 0.75)).
    """

    def __init__(
        self,
        group: Any,
        key: str,
        reduce_dims: Optional[Sequence[int]] = None,
        quantiles: Tuple[float, float] = (0.25, 0.5, 0.75),
    ) -> None:
        super().__init__()
        self.group = group
        self.key = key
        self.reduce_dims = tuple(reduce_dims) if reduce_dims is not None else ()
        self.quantiles = quantiles
        self.field_names = ["mean", "var", "min", "max"] + [
            f"q{int(q*100)}" for q in self.quantiles
        ]
        self.reset()

    def reset(self) -> None:
        """Initialize accumulators and buffers."""
        self.s: Optional[Tensor] = None
        self.s2: Optional[Tensor] = None
        self.min: Optional[Tensor] = None
        self.max: Optional[Tensor] = None
        self.count: int = 0
        self._buffer: List[Tensor] = []

    def execute(self) -> None:
        """Fetch tensor, optionally collapse dims, and update accumulators + buffer."""
        x = getattr(self.group, self.key, None)
        if x is None:
            x = self.group.states[self.key]
        # ensure tensor and detach+cpu
        if not torch.is_tensor(x):
            t = torch.tensor(x)
        else:
            t = x.detach()
        t = t.cpu()
        # collapse specified reduce_dims
        for d in sorted(self.reduce_dims, reverse=True):
            if t.dim() > d:
                t = t.mean(dim=d)
        # lazy init
        if self.s is None:
            self.s = torch.zeros_like(t)
            self.s2 = torch.zeros_like(t)
            self.min = t.clone()
            self.max = t.clone()
        # update accumulators
        self.s += t
        self.s2 += t * t
        self.min = torch.minimum(self.min, t)
        self.max = torch.maximum(self.max, t)
        self.count += 1
        self._buffer.append(t)

    def get_data(self) -> Dict[str, Tensor]:
        """Compute and return streaming statistics."""
        if self.s is None or self.count == 0:
            raise RuntimeError("No data collected: call execute() before get_data().")
        mean = self.s / self.count
        var = self.s2 / self.count - mean.pow(2)
        mn, mx = self.min, self.max
        allv = torch.stack(self._buffer, dim=0)
        qs = torch.quantile(allv, torch.tensor(self.quantiles), dim=0)
        out: Dict[str, Tensor] = {"mean": mean, "var": var, "min": mn, "max": mx}
        for qval, name in zip(qs.unbind(0), (f"q{int(q*100)}" for q in self.quantiles)):
            out[name] = qval
        return out
