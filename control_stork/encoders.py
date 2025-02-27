import torch
from typing import Tuple


class BaseEncoder(torch.nn.Module):
    """
    Base encoder class.
    All encoder implementations should inherit from this class.
    """

    def __init__(self):
        super().__init__()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input data.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")

    def compute_output_shape(self, input_shape: int) -> int:
        """
        Computes the output dimension given the input dimension.
        Must be implemented by subclasses.
        """
        raise NotImplementedError(
            "Subclasses must implement the compute_output_shape method."
        )

    def reset(self):
        """
        Reset any internal state. Default implementation does nothing.
        """
        pass


class EncoderStack(BaseEncoder):
    """
    EncoderStack applies a list of encoder objects sequentially.

    At initialization, it receives a list of encoders (subclasses of BaseEncoder).
    In the forward pass, it feeds the input through each encoder in order.
    The reset() method calls reset on all sub-encoders.
    The compute_output_shape method computes the overall output dimension
    by sequentially applying each encoder's compute_output_shape.
    """

    def __init__(self, encoders: list = []) -> None:
        """
        Args:
            encoders (list): A list of encoder objects (instances of subclasses of BaseEncoder).
        """
        super().__init__()
        # If the list is empty, create an IdentityEncoder as a placeholder.
        if not encoders:
            encoders = [IdentityEncoder()]
        # Wrap the list of encoders in a ModuleList to ensure proper registration.
        self.encoders = torch.nn.ModuleList(encoders)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Pass the input data through all encoders in sequence.

        Args:
            data (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The output after all encoders have processed the input.
        """
        for encoder in self.encoders:
            data = encoder(data)
        return data

    def reset(self):
        """
        Reset all sub-encoders.
        """
        for encoder in self.encoders:
            encoder.reset()

    def compute_output_shape(self, input_shape: int) -> int:
        """
        Compute the overall output shape by applying each encoder's compute_output_shape in sequence.

        Args:
            input_shape (int): The dimension of the input features.

        Returns:
            int: The dimension of the encoded output.
        """
        current_shape = input_shape
        for encoder in self.encoders:
            current_shape = encoder.compute_output_shape(current_shape)
        return current_shape


class FourierFeatureEncoder(BaseEncoder):
    """
    Encode input data using Fourier feature mapping.

    Given an input tensor of shape (batch_size, time_points, data_dims) with values in [-1, 1],
    the encoder computes sine and cosine features for each input element multiplied by a set
    of frequencies. The resulting encoded tensor has shape
        (batch_size, time_points, 2 * k * data_dims),
    where k is the number of frequencies per input dimension.
    """

    def __init__(self, k: int = 2) -> None:
        """
        Args:
            k (int): Number of frequencies per input dimension. Must be positive.
        """
        super().__init__()
        assert k > 0, "k must be positive"
        self.k: int = k

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        batch_size, time_points, data_dims = data.shape
        # Generate k frequencies linearly spaced between 1 and 2^(k-1) and multiply by π.
        frequencies = (
            torch.linspace(1.0, 2.0 ** (self.k - 1), self.k, device=data.device)
            * torch.pi
        )
        # Reshape frequencies to (1, 1, 1, k) for broadcasting.
        frequencies = frequencies.view(1, 1, 1, self.k)
        data_expanded = data.unsqueeze(-1)  # (batch_size, time_points, data_dims, 1)
        data_scaled = (
            data_expanded * frequencies
        )  # (batch_size, time_points, data_dims, k)
        sin_features = torch.sin(data_scaled)
        cos_features = torch.cos(data_scaled)
        # Concatenate sine and cosine along the last dimension.
        fourier_features = torch.cat([sin_features, cos_features], dim=-1)
        output = fourier_features.view(batch_size, time_points, -1)
        return output

    def compute_output_shape(self, input_size: int) -> int:
        return 2 * self.k * input_size


class RBFEncoder(BaseEncoder):
    """
    Encode input data using a Radial Basis Function (RBF) mapping.

    Given an input tensor of shape (batch_size, time_points, data_dims) with values
    assumed to be normalized (e.g. in [-1, 1]), this encoder computes a set of Gaussian
    RBF activations for each input dimension. Each dimension is encoded into a vector of length
    `num_centers` and the outputs are concatenated, resulting in an output of shape:
        (batch_size, time_points, data_dims * num_centers)

    The RBF activation is computed as:
        φ(x) = exp( - (x - μ)² / (2 * σ²) )
    where μ is one of the fixed centers and σ is the standard deviation.
    """

    def __init__(
        self,
        num_centers: int = 10,
        sigma: float = None,
        center_range: Tuple[float, float] = (-1.0, 1.0),
        sigma_scale: float = 1.0,
        learn_sigma: bool = False,
    ) -> None:
        """
        Args:
            num_centers (int): Number of RBF centers per input dimension.
            sigma (float): Standard deviation of the Gaussian. If None, set to spacing between centers.
            center_range (Tuple[float, float]): The range (min, max) over which to place centers.
            sigma_scale (float): Multiplicative factor for sigma.
            learn_sigma (bool): If True, sigma is a learnable parameter (stored in log-scale).
        """
        super().__init__()
        assert num_centers > 0, "num_centers must be positive"
        self.num_centers: int = num_centers

        default_sigma = 1.0
        if num_centers > 1:
            default_sigma = (center_range[1] - center_range[0]) / (num_centers - 1)
        sigma_val = sigma if sigma is not None else default_sigma
        sigma_val *= sigma_scale

        # Store sigma in log-domain for stability.
        log_sigma_val = torch.log(torch.tensor(sigma_val, dtype=torch.float32))
        if learn_sigma:
            self.log_sigma = torch.nn.Parameter(log_sigma_val)
        else:
            self.register_buffer("log_sigma", log_sigma_val)

        # Precompute centers and register as buffer.
        centers = torch.linspace(center_range[0], center_range[1], num_centers)
        self.register_buffer("centers", centers)

    @property
    def sigma(self) -> torch.Tensor:
        return torch.exp(self.log_sigma)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        batch_size, time_points, data_dims = data.shape
        data_expanded = data.unsqueeze(-1)  # (batch_size, time_points, data_dims, 1)
        centers = self.centers.view(1, 1, 1, self.num_centers)
        diff = data_expanded - centers
        rbf_features = torch.exp(-(diff**2) / (2 * (self.sigma**2)))
        output = rbf_features.view(batch_size, time_points, -1)
        return output

    def compute_output_shape(self, input_shape: int) -> int:
        return input_shape * self.num_centers


class LinearEncoder(BaseEncoder):
    """
    Encode input data using a linear transformation.

    Given an input tensor of shape (batch_size, time_points, data_dims) with values in [-1, 1],
    this encoder computes two encodings per input element:
        y1 = 0.5 * x + 0.5
        y2 = -0.5 * x + 0.5
    The outputs are concatenated along the last dimension, yielding a tensor of shape:
        (batch_size, time_points, data_dims * 2)
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        y1 = 0.5 * data + 0.5
        y2 = -0.5 * data + 0.5
        output = torch.cat([y1, y2], dim=-1)
        return output

    def compute_output_shape(self, input_shape: int) -> int:
        return input_shape * 2


class Linear4DEncoder(BaseEncoder):
    """
    A linear encoder that maps each input value x (assumed in a range, e.g. [-1,1])
    to four outputs using:
        y1 = a * x + b,
        y2 = a * x - b,
        y3 = -a * x + b,
        y4 = -a * x - b.
    Optionally, the outputs are normalized so that each 4D vector has unit L2 norm.

    Parameters:
      - a (float): scaling factor (default 0.5)
      - b (float): offset (default 0.5)
      - learn_params (bool): if True, a and b are learnable (stored in log scale).
      - normalize (bool): if True, normalize the 4D output.
    """

    def __init__(
        self,
        a: float = 0.5,
        b: float = 0.5,
        learn_params: bool = False,
        normalize: bool = False,
    ) -> None:
        super().__init__()
        self.normalize = normalize
        self.learn_params = learn_params

        if learn_params:
            self.log_a = torch.nn.Parameter(
                torch.log(torch.tensor(a, dtype=torch.float32))
            )
            self.log_b = torch.nn.Parameter(
                torch.log(torch.tensor(b, dtype=torch.float32))
            )
        else:
            self.register_buffer("a_val", torch.tensor(a, dtype=torch.float32))
            self.register_buffer("b_val", torch.tensor(b, dtype=torch.float32))

    @property
    def a(self) -> torch.Tensor:
        if self.learn_params:
            return torch.exp(self.log_a)
        else:
            return self.a_val

    @property
    def b(self) -> torch.Tensor:
        if self.learn_params:
            return torch.exp(self.log_b)
        else:
            return self.b_val

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        y1 = self.a * data + self.b
        y2 = self.a * data - self.b
        y3 = -self.a * data + self.b
        y4 = -self.a * data - self.b
        out = torch.stack(
            [y1, y2, y3, y4], dim=-1
        )  # Shape: (batch, time, data_dims, 4)
        if self.normalize:
            norm = torch.norm(out, dim=-1, keepdim=True)
            out = out / (norm + 1e-8)
        batch_size, time_points, data_dims, _ = out.shape
        out = out.view(batch_size, time_points, data_dims * 4)
        return out

    def compute_output_shape(self, input_shape: int) -> int:
        return input_shape * 4


class DeltaEncoder(BaseEncoder):
    """
    Delta Encoder that keeps a copy of the previous input in its state.
    In the forward pass, it computes the difference between the current input and the previous input,
    scales this difference by a factor diff_scale, and then concatenates the scaled difference to the
    current input along the last dimension. This doubles the output dimension relative to the input.

    After reset(), the first forward call will produce a zero difference.

    Args:
        diff_scale (float): Scaling factor for the computed difference (default: 1.0).
    """

    def __init__(self, diff_scale: float = 1.0) -> None:
        super().__init__()
        self.last_input = None
        self.diff_scale = diff_scale

    def reset(self):
        """Reset the stored previous input."""
        self.last_input = None

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # Use clone() to avoid unintended in-place modifications.
        if self.last_input is None:
            diff = torch.zeros_like(data)
        else:
            diff = data - self.last_input
        # Store a clone of the current input for the next forward call.
        self.last_input = data.clone()
        # Scale the difference.
        diff = self.diff_scale * diff
        return torch.cat([data, diff], dim=-1)

    def compute_output_shape(self, input_shape: int) -> int:
        return input_shape * 2


class IdentityEncoder(BaseEncoder):
    """
    Identity encoder that simply returns the input data unchanged.
    This is useful for testing and as a placeholder for encoders that do not modify the input.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return data

    def compute_output_shape(self, input_shape: int) -> int:
        return input_shape