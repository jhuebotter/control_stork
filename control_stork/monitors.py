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
        self.data.append(self.group.states['out'].detach())

    def get_data(self) -> torch.Tensor:
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
        return torch.stack(self.data, dim=1).cpu()


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
        return self.data.cpu()

class PopulationSpikeCountMonitor(Monitor):
    """Counts total number of spikes (sum over time in get_out_sequence() for the group)

    Args:
        group: The group to record from

    Returns:
        A tensor with spike counts for each input and neuron
    """

    def __init__(self, group: CellGroup, per_example: bool = False) -> None:
        super().__init__()
        self.group = group
        self.per_example = per_example

    def reset(self) -> None:
        self.data = []

    def execute(self) -> None:
        self.data.append(self.group.states['out'].detach())

    def get_data(self) -> torch.Tensor:
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
        self.data.append(self.group.states['out'].detach())

    def get_data(self) -> torch.Tensor:
        s1 = torch.stack(self.data, dim=1).cpu()
        s1 = s1.reshape(s1.shape[0], s1.shape[1], self.group.nb_units)
        return torch.sum(s1, dim=-1) / self.group.nb_units


class MeanVarianceMonitor(Monitor):
    """Measures mean and variance of input

    Args:
        group: The group to record from
        state (string): State variable to monitor (Monitors mean and variance of a state variable)


    Returns:
        A tensors with mean and variance for each neuron/state along the last dim
    """

    def __init__(self, group: CellGroup, state: str = "input") -> None:
        super().__init__()
        self.group = group
        self.key = state

    def reset(self) -> None:
        self.s = 0
        self.s2 = 0
        self.c = 0

    def execute(self) -> None:
        tns = self.group.states[self.key]
        self.s += tns.detach()
        self.s2 += torch.square(tns).detach()
        self.c += 1

    def get_data(self) -> torch.Tensor:
        mean = self.s / self.c
        var = self.s2 / self.c - mean**2
        return torch.stack((mean, var), len(mean.shape)).cpu()


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
