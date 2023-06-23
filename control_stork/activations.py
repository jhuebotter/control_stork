import torch
import torch.nn as nn
import math


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        # This is a hack to make the class callable like other activation functions
        return self.apply(*args, **kwargs)


class SuperSpike(SurrogateSpike):
    """
    Autograd SuperSpike nonlinearity implementation.

    The steepness parameter beta can be accessed via the static member
    self.beta.
    """

    beta = 20.0
    gamma = 1.0  # gradient scale

    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the step function output. ctx is the context object
        that is used to stash information for backward pass computations.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        In the backward pass we receive a Tensor containing the gradient of the
        loss with respect to the output, and we compute the surrogate gradient
        of the loss with respect to the input. Here we assume the standardized
        negative part of a fast sigmoid as this was done in Zenke & Ganguli
        (2018).
        """
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input / (SuperSpike.beta * torch.abs(input) + 1.0) ** 2
        return grad * SuperSpike.gamma


class SuperSpike_MemClamp(SurrogateSpike):
    """
    Variant of SuperSpike with clamped membrane potential at 1.0
    """

    beta = 20.0
    gamma = 1.0  # gradient scale

    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the step function output. ctx is the context object
        that is used to stash information for backward pass computations.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        In the backward pass we receive a Tensor containing the gradient of the
        loss with respect to the output, and we compute the surrogate gradient
        of the loss with respect to the input. Here we assume the standardized
        negative part of a fast sigmoid as this was done in Zenke & Ganguli
        (2018).
        """
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = (
            grad_input
            / (SuperSpike_MemClamp.beta * torch.abs(torch.relu(-input)) + 1.0) ** 2
        )
        return grad * SuperSpike_MemClamp.gamma


class SuperSpike_rescaled(SurrogateSpike):
    """
    Version of SuperSpike where the gradient is re-scaled so that it equals one at
    resting membrane potential
    """

    beta = 20.0
    gamma = 1.0  # gradient scale

    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the step function output. ctx is the context object
        that is used to stash information for backward pass computations.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        In the backward pass we receive a Tensor containing the gradient of the
        loss with respect to the output, and we compute the surrogate gradient
        of the loss with respect to the input. Here we assume the standardized
        negative part of a fast sigmoid as this was done in Zenke & Ganguli
        (2018).
        """
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        rescale_val = 1 / ((SuperSpike_rescaled.beta + 1) ** 2)
        grad = (
            grad_input
            / (SuperSpike_rescaled.beta * torch.abs(input) + 1.0) ** 2
            / rescale_val
        )
        return grad * SuperSpike_rescaled.gamma


class MultiSpike(SurrogateSpike):
    """
    Autograd MultiSpike nonlinearity implementation.

    The steepness parameter beta can be accessed via the static member
    self.beta (default=100).
    """

    beta = 100.0
    maxspk = 10.0
    gamma = 1.0  # gradient scale


    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the step function output. ctx is the context object
        that is used to stash information for backward pass computations.
        """
        ctx.save_for_backward(input)
        out = nn.functional.hardtanh(torch.round(input + 0.5), 0.0, MultiSpike.maxspk)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        In the backward pass we receive a Tensor containing the gradient of the
        loss with respect to the output, and we compute the surrogate gradient
        of the loss with respect to the input. Here we assume the standardized
        negative part of a fast sigmoid as this was done in Zenke & Ganguli
        (2018).
        """
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = (
            grad_input
            / (
                MultiSpike.beta * torch.abs(input - torch.relu(torch.round(input)))
                + 1.0
            )
            ** 2
        )
        return grad * MultiSpike.gamma


class SuperSpike_asymptote(SurrogateSpike):
    """
    Autograd SuperSpike nonlinearity implementation with asymptotic behavior of step.

    The steepness parameter beta can be accessed via the static member
    self.beta (default=100).
    """

    beta = 100.0
    gamma = 1.0  # gradient scale

    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the step function output. ctx is the context object
        that is used to stash information for backward pass computations.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        In the backward pass we receive a Tensor containing the gradient of the
        loss with respect to the output, and we compute the surrogate gradient
        of the loss with respect to the input. Here we assume the standardized
        negative part of a fast sigmoid as this was done in Zenke & Ganguli
        (2018).
        """
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = (
            SuperSpike_asymptote.beta
            * grad_input
            / (SuperSpike_asymptote.beta * torch.abs(input) + 1.0) ** 2
        )
        return grad * SuperSpike_asymptote.gamma


class TanhSpike(SurrogateSpike):
    """
    Autograd Tanh et al. nonlinearity implementation.

    The steepness parameter beta can be accessed via the static member
    self.beta (default=100).
    """

    beta = 100.0
    gamma = 1.0  # gradient scale


    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the step function output. ctx is the context object
        that is used to stash information for backward pass computations.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        In the backward pass we receive a Tensor containing the gradient of the
        loss with respect to the output, and we compute the surrogate gradient
        of the loss with respect to the input. Here we assume the standardized
        negative part of a fast sigmoid as this was done in Zenke & Ganguli
        (2018).
        """
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        beta = TanhSpike.beta
        grad = grad_input * (1.0 + (1.0 - torch.tanh(input * beta) ** 2))
        return grad * TanhSpike.gamma


class SigmoidSpike(SurrogateSpike):
    """
    Autograd surrogate gradient nonlinearity implementation which uses the derivative of a sigmoid in the backward pass.

    The steepness parameter beta can be accessed via the static member self.beta (default=100).
    """

    beta = 100.0
    gamma = 1.0  # gradient scale


    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the step function output. ctx is the context object
        that is used to stash information for backward pass computations.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        In the backward pass we receive a Tensor containing the gradient of the
        loss with respect to the output, and we compute the surrogate gradient
        of the loss with respect to the input. Here we assume the standardized
        negative part of a fast sigmoid as this was done in Zenke & Ganguli
        (2018).
        """
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        sig = torch.sigmoid(SigmoidSpike.beta * input)
        dsig = sig * (1.0 - sig)
        grad = grad_input * dsig
        return grad * SigmoidSpike.gamma


class EsserSpike(SurrogateSpike):
    """
    Autograd surrogate gradient nonlinearity implementation which uses piecewise linear pseudo derivative in the backward pass as suggested in:

        Esser, S.K., Merolla, P.A., Arthur, J.V., Cassidy, A.S., Appuswamy, R.,
        Andreopoulos, A., Berg, D.J., McKinstry, J.L., Melano, T., Barch, D.R.,
        et al. (2016). Convolutional networks for fast, energy-efficient
        neuromorphic computing. Proc Natl Acad Sci U S A 113, 11441–11446.
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5068316/

    The steepness parameter beta can be accessed via the static member self.beta (default=1.0).
    """

    beta = 1.0
    gamma = 1.0  # gradient scale


    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the step function output. ctx is the context object
        that is used to stash information for backward pass computations.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        In the backward pass we receive a Tensor containing the gradient of the
        loss with respect to the output, and we compute the surrogate gradient
        of the loss with respect to the input. Here we assume the standardized
        negative part of a fast sigmoid as this was done in Zenke & Ganguli
        (2018).
        """
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input * torch.max(
            torch.zeros_like(input), 1.0 - torch.abs(EsserSpike.beta * input)
        )
        return grad * EsserSpike.gamma


class HardTanhSpike(SurrogateSpike):
    """
    Autograd Esser et al. nonlinearity implementation.

    The steepness parameter beta can be accessed via the static member
    self.beta (default=100).
    """

    beta = 100.0
    gamma = 1.0  # gradient scale


    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the step function output. ctx is the context object
        that is used to stash information for backward pass computations.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        In the backward pass we receive a Tensor containing the gradient of the
        loss with respect to the output, and we compute the surrogate gradient
        of the loss with respect to the input. Here we assume the standardized
        negative part of a fast sigmoid as this was done in Zenke & Ganguli
        (2018).
        """
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        beta = HardTanhSpike.beta
        grad = grad_input * (1.0 + torch.nn.functional.hardtanh(input * beta))
        return grad * HardTanhSpike.gamma


class SuperSpike_norm(SurrogateSpike):
    """
    Autograd SuperSpike nonlinearity implementation.

    The steepness parameter beta can be accessed via the static member
    self.beta (default=100).
    """

    beta = 100.0
    xi = 1e-2
    gamma = 1.0  # gradient scale


    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the step function output. ctx is the context object
        that is used to stash information for backward pass computations.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        In the backward pass we receive a Tensor containing the gradient of the
        loss with respect to the output, and we compute the surrogate gradient
        of the loss with respect to the input. Here we assume the standardized
        negative part of a fast sigmoid as this was done in Zenke & Ganguli
        (2018).
        """
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input / (SuperSpike_norm.beta * torch.abs(input) + 1.0) ** 2
        # standardize gradient
        standard_grad = grad / (
            SuperSpike_norm.xi + torch.norm(torch.mean(grad, dim=0))
        )
        return standard_grad * SuperSpike_norm.gamma


def gaussian(x, mu=0.0, sigma=0.5):
    return (
        torch.exp(-((x - mu) ** 2) / (2 * sigma**2))
        / torch.sqrt(2 * torch.tensor(math.pi))
        / sigma
    )


class GaussianSpike(SurrogateSpike):
    """
    Autograd Gaussian nonlinearity implementation.

    """

    gamma = 0.5  # gradient scale
    lens = 0.3
    scale = 6.0
    hight = 0.15

    @staticmethod
    def forward(ctx, input):  # input = membrane potential- threshold
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the step function output. ctx is the context object
        that is used to stash information for backward pass computations.
        """
        ctx.save_for_backward(input)
        return input.gt(0).float()  # is firing ???

    @staticmethod
    def backward(ctx, grad_output):  # approximate the gradients
        """
        In the backward pass we receive a Tensor containing the gradient of the
        loss with respect to the output, and we compute the surrogate gradient
        of the loss with respect to the input. Here we assume the standardized
        negative part of a fast sigmoid as this was done in Yin, Corradi, and
        Bothe (2021).
        """
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()

        # temp =  gaussian(input, mu=0., sigma=GaussianSpike.lens)
        # temp = torch.exp(-(input**2)/(2*GaussianSpike.lens**2))/torch.sqrt(2*torch.tensor(math.pi))/GaussianSpike.lens

        temp = (
            gaussian(input, mu=0.0, sigma=GaussianSpike.lens)
            * (1.0 + GaussianSpike.hight)
            - gaussian(
                input,
                mu=GaussianSpike.lens,
                sigma=GaussianSpike.scale * GaussianSpike.lens,
            )
            * GaussianSpike.hight
            - gaussian(
                input,
                mu=-GaussianSpike.lens,
                sigma=GaussianSpike.scale * GaussianSpike.lens,
            )
            * GaussianSpike.hight
        )
        return grad_input * temp.float() * GaussianSpike.gamma
