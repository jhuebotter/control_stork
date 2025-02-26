import torch
import torch.nn as nn

from torch.nn.parameter import Parameter

import numpy as np

from . import core
from . import constraints as stork_constraints
from .extratypes import *

# from . import nodes
# from . import initializers

# TODO: complete docstrings
# TODO: add type hints without circular imports


class BaseConnection(core.NetworkNode):
    def __init__(
        self,
        src,  #: nodes.CellGroup,
        dst,  #: nodes.CellGroup,
        target: Optional[str] = None,
        name: Optional[str] = None,
        regularizers: Optional[Iterable] = None,
        constraints: Optional[Iterable] = None,
    ) -> None:
        """Abstract base class of Connection objects.

        Args:
            src (CellGroup): The source group
            dst (CellGroup): The destination group
            target (string, optional): The name of the target state tensor.
            name (string, optional): Name of the node
            regularizers (list): List of regularizer objects.
            constraints (list): List of constraints.

        """

        super(BaseConnection, self).__init__(name=name, regularizers=regularizers)
        self.src = src
        self.dst = dst

        self.recurrent = src == dst

        if target is None:
            self.target = dst.default_target
        else:
            self.target = target

        if constraints is None:
            self.constraints = []
        elif type(constraints) == list:
            self.constraints = constraints
        elif issubclass(type(constraints), stork_constraints.WeightConstraint):
            self.constraints = [constraints]
        else:
            raise ValueError

    def init_parameters(self, initializer) -> None:
        """
        Initializes connection weights and biases.
        """
        initializer.initialize(self)
        self.apply_constraints()

    def propagate(self):
        raise NotImplementedError

    def apply_constraints(self):
        raise NotImplementedError


class Connection(BaseConnection):
    def __init__(
        self,
        src,  #: nodes.CellGroup,
        dst,  #: nodes.CellGroup,
        operation: nn.Module = nn.Linear,
        target: Optional[str] = None,
        bias: bool = False,
        requires_grad: bool = True,
        propagate_gradients: bool = True,
        flatten_input: bool = False,
        name: Optional[str] = None,
        regularizers: Optional[Iterable] = None,
        constraints: Optional[Iterable] = None,
        **kwargs
    ) -> None:
        super(Connection, self).__init__(
            src,
            dst,
            name=name,
            target=target,
            regularizers=regularizers,
            constraints=constraints,
        )

        self.requires_grad = requires_grad
        self.propagate_gradients = propagate_gradients
        self.flatten_input = flatten_input

        if flatten_input:
            self.op = operation(src.nb_units, dst.shape[0], bias=bias, **kwargs)
        else:
            self.op = operation(src.shape[0], dst.shape[0], bias=bias, **kwargs)
        for param in self.op.parameters():
            param.requires_grad = requires_grad

    def add_diagonal_structure(self, width: float = 1.0, ampl: float = 1.0) -> None:
        if type(self.op) != nn.Linear:
            raise ValueError("Expected op to be nn.Linear to add diagonal structure.")
        A = np.zeros(self.op.weight.shape)
        x = np.linspace(0, A.shape[0], A.shape[1])
        for i in range(len(A)):
            A[i] = ampl * np.exp(-((x - i) ** 2) / width**2)
        self.op.weight.data += torch.from_numpy(A)

    def get_weights(self) -> torch.Tensor:
        return self.op.weight
    
    def get_bias(self) -> torch.Tensor:
        return self.op.bias
    
    def get_weight_regularizer_loss(self, reduction="mean") -> torch.Tensor:
        reg_loss = torch.tensor(0.0, device=self.device)
        for reg in self.regularizers:
            reg_loss = reg_loss + reg(self.get_weights(), reduction=reduction)
        return reg_loss
    
    def get_bias_regularizer_loss(self, reduction="mean") -> torch.Tensor:
        reg_loss = torch.tensor(0.0, device=self.device)
        bias = self.get_bias()
        if bias is not None:
            for reg in self.regularizers:
                reg_loss = reg_loss + reg(bias, reduction=reduction)
        return reg_loss
    
    def get_weight_regularizer_grad(self) -> torch.Tensor:
        reg_grad = torch.tensor(0.0, device=self.device)
        for reg in self.regularizers:
            reg_grad = reg_grad + reg.grad(self.get_weights())
        return reg_grad
    
    def get_bias_regularizer_grad(self) -> torch.Tensor:
        reg_grad = torch.tensor(0.0, device=self.device)
        bias = self.get_bias()
        if bias is not None:
            for reg in self.regularizers:
                reg_grad = reg_grad + reg.grad(bias)
        return reg_grad

    def get_regularizer_loss(self) -> torch.Tensor:
        """this always uses mean reduction so it can be added to other losses"""
        reg_loss = torch.tensor(0.0, device=self.device)
        for reg in self.regularizers:
            reg_loss += reg(self.get_weights())
            bias = self.get_bias()
            if bias is not None:
                reg_loss = reg_loss + reg(bias)
        return reg_loss

    def forward(self) -> None:
        preact = self.src.out
        if not self.propagate_gradients:
            preact = preact.detach()
        if self.flatten_input:
            shp = preact.shape
            preact = preact.reshape(shp[:1] + (-1,))

        out = self.op(preact)
        self.dst.add_to_state(self.target, out)

    def propagate(self) -> None:
        self.forward()

    def apply_constraints(self) -> None:
        for const in self.constraints:
            const.apply(self.op.weight)


class BottleneckLinearConnection(BaseConnection):
    def __init__(
        self,
        src,  #: nodes.CellGroup,
        dst,  #: nodes.CellGroup,
        target: Optional[str] = None,
        bias: bool = False,
        latent_bias: bool = False,
        n_dims: int = 1,
        requires_grad: bool = True,
        propagate_gradients: bool = True,
        flatten_input: bool = False,
        name: Optional[str] = None,
        regularizers: Optional[Iterable] = None,
        constraints: Optional[Iterable] = None,
        **kwargs
    ) -> None:
        super(BottleneckLinearConnection, self).__init__(
            src,
            dst,
            name=name,
            target=target,
            regularizers=regularizers,
            constraints=constraints,
        )

        self.requires_grad = requires_grad
        self.propagate_gradients = propagate_gradients
        self.flatten_input = flatten_input
        self.n_dims = n_dims

        # TODO: ideally combine pre_op and op into one layer

        self.pre_op = nn.Linear(
            src.nb_units if flatten_input else src.shape[0],
            self.n_dims,
            bias=latent_bias,
            **kwargs
        )
        self.op = nn.Linear(self.n_dims, dst.shape[0], bias=bias, **kwargs)

        for param in self.pre_op.parameters():
            param.requires_grad = requires_grad
        for param in self.op.parameters():
            param.requires_grad = requires_grad

    def add_diagonal_structure(self, width: float = 1.0, ampl: float = 1.0) -> None:
        if type(self.op) != nn.Linear:
            raise ValueError("Expected op to be nn.Linear to add diagonal structure.")
        A = np.zeros(self.op.weight.shape)
        x = np.linspace(0, A.shape[0], A.shape[1])
        for i in range(len(A)):
            A[i] = ampl * np.exp(-((x - i) ** 2) / width**2)
        self.op.weight.data += torch.from_numpy(A)

    def get_weights(self) -> (torch.Tensor, torch.Tensor):
        return self.pre_op.weight, self.op.weight
    
    def get_bias(self) -> (torch.Tensor, torch.Tensor):
        return self.pre_op.bias, self.op.bias

    def get_regularizer_loss(self) -> torch.Tensor:
        reg_loss = torch.tensor(0.0, device=self.device)
        for reg in self.regularizers:
            for l in [self.op, self.pre_op]:
                reg_loss += reg(l.weight)
                if l.bias is not None:
                    reg_loss += reg(l.bias)
        return reg_loss

    def forward(self) -> None:
        preact = self.src.out
        if not self.propagate_gradients:
            preact = preact.detach()
        if self.flatten_input:
            shp = preact.shape
            preact = preact.reshape(shp[:1] + (-1,))

        latent = self.pre_op(preact)
        out = self.op(latent)
        self.dst.add_to_state(self.target, out)

    def propagate(self) -> None:
        self.forward()

    def apply_constraints(self) -> None:
        for const in self.constraints:
            const.apply(self.op.weight)


class IdentityConnection(BaseConnection):
    def __init__(
        self,
        src,  #: nodes.CellGroup,
        dst,  #: nodes.CellGroup,
        target: Optional[str] = None,
        bias: bool = False,
        requires_grad: bool = True,
        name: Optional[str] = None,
        regularizers: Optional[Iterable] = None,
        constraints: Optional[Iterable] = None,
        tie_weights: Optional[Iterable] = None,
        weight_scale: float = 1.0,
    ) -> None:
        """Initialize IdentityConnection

        Args:
            tie_weights (list of int, optional): Tie weights along dims given in list
            weight_scale (float, optional): Scale everything by this factor. Useful when the connection is used for relaying currents rather than spikes.
        """
        super(IdentityConnection, self).__init__(
            src,
            dst,
            name=name,
            target=target,
            regularizers=regularizers,
            constraints=constraints,
        )

        self.requires_grad = requires_grad
        self.weight_scale = weight_scale
        wshp = src.shape

        # Set weights tensor dimension to 1 along tied dimensions
        if tie_weights is not None:
            wshp = list(wshp)
            for d in tie_weights:
                wshp[d] = 1
            wshp = tuple(wshp)

        self.weights = Parameter(torch.randn(wshp), requires_grad=requires_grad)
        if bias:
            self.bias = Parameter(torch.randn(wshp), requires_grad=requires_grad)

    def get_weights(self) -> torch.Tensor:
        return self.weights

    def get_regularizer_loss(self) -> torch.Tensor:
        reg_loss = torch.tensor(0.0, device=self.device)
        for reg in self.regularizers:
            reg_loss += reg(self.get_weights())
        return reg_loss

    def apply_constraints(self) -> None:
        for const in self.constraints:
            const.apply(self.weights)

    def forward(self) -> None:
        preact = self.src.out
        if self.bias is None:
            self.dst.scale_and_add_to_state(
                self.weight_scale, self.target, self.weights * preact
            )
        else:
            self.dst.scale_and_add_to_state(
                self.weight_scale, self.target, self.weights * preact + self.bias
            )

    def propagate(self) -> None:
        self.forward()


class ConvConnection(Connection):
    def __init__(
        self,
        src,  #: nodes.CellGroup,
        dst,  #: nodes.CellGroup,
        conv: object = nn.Conv1d,
        **kwargs
    ) -> None:
        super(ConvConnection, self).__init__(src, dst, operation=conv, **kwargs)


class Conv2dConnection(Connection):
    def __init__(
        self,
        src,  #: nodes.CellGroup,
        dst,  #: nodes.CellGroup,
        conv: object = nn.Conv2d,
        **kwargs
    ) -> None:
        super(Conv2dConnection, self).__init__(src, dst, operation=conv, **kwargs)
