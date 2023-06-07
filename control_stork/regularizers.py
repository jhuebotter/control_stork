import torch
from extratypes import *

class ActivityRegularizer:
    """Abstract base class for activity regularizers."""

    def __init__(
        self, strength: float = 1.0, threshold: float = 0.0, dims: Optional[Union[int, tuple, list]] = -1
    ) -> None:
        """Constructor

        Args:
            strength (float, optional): Regularizer strengh. Defaults to 1.0.
            threshold (float, optional): Upper threshold (in number of spikes, not firing rate). Defaults to 0.0.
            dims (int, optional): The dimensions to average spikes/activity excluding time dimension.
                                  Defaults to -1, which supports fully connected networks and 1D-Conv nets.
                                  For 2D-Conv nets, set to dims=(-2,-1) or dims=(3,4) (equivalent).
                                  To implement a per-neuron regularizer, set dims=None or 0 to average over the batch.
        """

        self.strength = float(strength)
        self.threshold = float(threshold)
        self.dims = dims

        # Assert that dimensions is either False, int, tuple or list
        if self.dims is not None:
            assert isinstance(self.dims, (int, tuple, list))

    def __call__(self, group) -> torch.Tensor:
        """Expects input with (batch x time x units)"""
        act = group.get_out_sequence()  # get output
        # cnt = torch.sum(act, dim=1)  # get spikecount
        cnt = torch.mean(act, dim=1)  # get average "firing rate"

        # if population-level regularizer, calculate mean across defined dims
        if self.dims is not None:
            cnt = torch.mean(cnt, dim=self.dims)

        return self.calc_regloss(cnt)

    def calc_regloss(self, cnt: torch.Tensor) -> torch.Tensor:
        """
        Args: cnt:    Spikecount
        """
        raise NotImplementedError("Abstract base class.")


class ActivityRegularizerL1(ActivityRegularizer):
    """Penalizes activity above and below threshold"""

    def calc_regloss(self, cnt: torch.Tensor) -> torch.Tensor:
        reg = cnt - self.threshold
        reg_loss = self.strength * torch.mean(torch.abs(reg))
        return reg_loss


class UpperBoundL1(ActivityRegularizer):
    """Provides an upper bound L1 regularizer on the spike count"""

    def calc_regloss(self, cnt: torch.Tensor) -> torch.Tensor:
        reg = torch.relu(cnt - self.threshold)
        reg_loss = self.strength * torch.mean(torch.abs(reg))
        return reg_loss


class LowerBoundL1(ActivityRegularizer):
    """Provides a lower bound L1 regularizer on the spike count"""

    def calc_regloss(self, cnt: torch.Tensor) -> torch.Tensor:
        reg = torch.relu(-(cnt - self.threshold))
        reg_loss = self.strength * torch.mean(torch.abs(reg))
        return reg_loss


class UpperBoundL2(ActivityRegularizer):
    """Provides an upper bound L2 regularizer on the spike count"""

    def calc_regloss(self, cnt: torch.Tensor) -> torch.Tensor:
        reg = torch.relu(cnt - self.threshold)
        reg_loss = self.strength * torch.mean(torch.square(reg))
        return reg_loss


class LowerBoundL2(ActivityRegularizer):
    """Provides a lower bound L2 regularizer on the spike count"""

    def calc_regloss(self, cnt: torch.Tensor) -> torch.Tensor:
        reg = torch.relu(-(cnt - self.threshold))
        reg_loss = self.strength * torch.mean(torch.square(reg))
        return reg_loss


class WeightL2Regularizer:
    """A mean square target rate regularizer"""

    def __init__(self, strength: float = 1.0) -> None:
        """Constructor

        Args:
            strength: regularizer strengh
        """
        self.strength = float(strength)

    def __call__(self, w: torch.Tensor) -> torch.Tensor:
        """Expects input with weights (channels x stuff)"""
        return self.strength * torch.mean(w**2)


class WeightL1Regularizer:
    """A mean square target rate regularizer"""

    def __init__(self, strength: float = 1.0):
        """Constructor

        Args:
            strength: regularizer strengh
        """
        self.strength = float(strength)

    def __call__(self, w: torch.Tensor) -> torch.Tensor:
        """Expects input with weights (channels x stuff)"""
        return self.strength * torch.mean(torch.abs(w))
