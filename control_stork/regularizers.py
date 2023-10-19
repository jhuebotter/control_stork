import torch
from .extratypes import *


class ActivityRegularizer:
    """Abstract base class for activity regularizers."""

    def __init__(
        self, strength: float = 1.0, threshold: float = 0.0, dims: Optional[Union[int, tuple, list]] = -1, basis: str = "out"
    ) -> None:
        """Constructor

        Args:
            strength (float, optional): Regularizer strengh. Defaults to 1.0.
            threshold (float, optional): Upper threshold (in number of spikes, not firing rate). Defaults to 0.0.
            dims (int, optional): The dimensions to average spikes/activity excluding time dimension.
                                  Defaults to -1, which supports fully connected networks and 1D-Conv nets.
                                  For 2D-Conv nets, set to dims=(-2,-1) or dims=(3,4) (equivalent).
                                  To implement a per-neuron regularizer, set dims=None or 0 to average over the batch.
            basis (str, optional): The basis to use for the regularizer. Defaults to "out".
        """

        self.strength = float(strength)
        self.threshold = float(threshold)
        self.dims = dims
        self.basis = basis

        # Assert that dimensions is either False, int, tuple or list
        if self.dims is not None:
            assert isinstance(self.dims, (int, tuple, list))

    def __call__(self, group, reduction: str = "mean") -> torch.Tensor:
        """Expects input with (batch x time x units)"""
        if self.basis not in group.store_state_sequences:
            raise ValueError(
                f"Regularizer basis {self.basis} not stored in group {group.name}."
            )
        act = group.get_state_sequence(self.basis)  # get sequence used for regularization
        # act = group.get_out_sequence()  # get output
        # cnt = torch.sum(act, dim=1)  # get spikecount
        avg = torch.mean(act, dim=1)  # get average "spike density"

        # if population-level regularizer, calculate mean across defined dims
        if self.dims is not None:
            avg = torch.mean(avg, dim=self.dims)

        return self.calc_regloss(avg, reduction=reduction)

    def calc_regloss(self, cnt: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        """
        Args: cnt:    Spikecount
        """
        raise NotImplementedError("Abstract base class.")


class ActivityRegularizerL1(ActivityRegularizer):
    """Penalizes activity above and below threshold"""

    def calc_regloss(self, cnt: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        reg = cnt - self.threshold
        reg = torch.abs(reg)
        if reduction == "mean":
            reg_loss = torch.mean(reg)
        elif reduction == "sum":
            reg_loss = torch.sum(reg)
        elif reduction == "none":
            # this will return a loss per batch element
            reg_loss = reg.view(reg.size(0), -1).mean(dim=1)
        return self.strength * reg_loss


class ActivityRegularizerL2(ActivityRegularizer):
    """Penalizes square of activity above and below threshold"""

    def calc_regloss(self, cnt: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        reg = cnt - self.threshold
        reg = torch.square(reg)
        if reduction == "mean":
            reg_loss = torch.mean(reg)
        elif reduction == "sum":
            reg_loss = torch.sum(reg)
        elif reduction == "none":
            # this will return a loss per batch element
            reg_loss = reg.view(reg.size(0), -1).mean(dim=1)
        return self.strength * reg_loss


class UpperBoundL1(ActivityRegularizer):
    """Provides an upper bound L1 regularizer on the spike count"""

    def calc_regloss(self, cnt: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        reg = torch.relu(cnt - self.threshold)
        reg = torch.abs(reg)
        if reduction == "mean":
            reg_loss = torch.mean(reg)
        elif reduction == "sum":
            reg_loss = torch.sum(reg)
        elif reduction == "none":
            # this will return a loss per batch element
            reg_loss = reg.view(reg.size(0), -1).mean(dim=1)
        return self.strength * reg_loss


class LowerBoundL1(ActivityRegularizer):
    """Provides a lower bound L1 regularizer on the spike count"""

    def calc_regloss(self, cnt: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        reg = torch.relu(-(cnt - self.threshold))
        reg = torch.abs(reg)
        if reduction == "mean":
            reg_loss = torch.mean(reg)
        elif reduction == "sum":
            reg_loss = torch.sum(reg)
        elif reduction == "none":
            # this will return a loss per batch element
            reg_loss = reg.view(reg.size(0), -1).mean(dim=1)
        return self.strength * reg_loss


class UpperBoundL2(ActivityRegularizer):
    """Provides an upper bound L2 regularizer on the spike count"""

    def calc_regloss(self, cnt: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        reg = torch.relu(cnt - self.threshold)
        reg = torch.square(reg)
        if reduction == "mean":
            reg_loss = torch.mean(reg)
        elif reduction == "sum":
            reg_loss = torch.sum(reg)
        elif reduction == "none":
            # this will return a loss per batch element
            reg_loss = reg.view(reg.size(0), -1).mean(dim=1)
        return self.strength * reg_loss


class LowerBoundL2(ActivityRegularizer):
    """Provides a lower bound L2 regularizer on the spike count"""

    def calc_regloss(self, cnt: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        reg = torch.relu(-(cnt - self.threshold))
        reg = torch.square(reg)
        if reduction == "mean":
            reg_loss = torch.mean(reg)
        elif reduction == "sum":
            reg_loss = torch.sum(reg)
        elif reduction == "none":
            # this will return a loss per batch element
            reg_loss = reg.view(reg.size(0), -1).mean(dim=1)
        return self.strength * reg_loss


class WeightRegularizer:
    """Abstract base class for weight regularizers."""

    def __init__(self, strength: float = 1.0) -> None:
        """Constructor

        Args:
            strength: regularizer strength
        """
        self.strength = float(strength)

    def __call__(self, w: torch.Tensor, reduction="mean") -> torch.Tensor:
        """Expects input with weights (channels x stuff)"""
        raise NotImplementedError("Abstract base class.")


class WeightL2Regularizer(WeightRegularizer):
    """A mean square target rate regularizer"""

    def __call__(self, w: torch.Tensor, reduction="mean") -> torch.Tensor:
        """Expects input with weights (channels x stuff)"""
        if reduction == "mean":
            return self.strength * torch.mean(w.square())
        elif reduction == "sum":
            return self.strength * torch.sum(w.square())
        elif reduction == "none":
            # this will return a loss per weight element
            return self.strength * (w.square())
        
    def grad(self, w: torch.Tensor) -> torch.Tensor:
        """Expects input with weights (channels x stuff)"""
        return self.strength * 2 * w


class WeightL1Regularizer(WeightRegularizer):
    """A mean square target rate regularizer"""

    def __call__(self, w: torch.Tensor, reduction="mean") -> torch.Tensor:
        """Expects input with weights (channels x stuff)"""
        if reduction == "mean":
            return self.strength * torch.mean(w.abs())
        elif reduction == "sum":
            return self.strength * torch.sum(w.abs())
        elif reduction == "none":
            # this will return a loss per weight element
            return self.strength * (w.abs())
        
    def grad(self, w: torch.Tensor) -> torch.Tensor:
        """Expects input with weights (channels x stuff)"""
        return self.strength * torch.sign(w)
