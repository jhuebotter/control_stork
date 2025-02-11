import torch
from typing import Tuple


class FourierFeatureEncoder(torch.nn.Module):
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
        Initialize the FourierFeatureEncoder.

        Args:
            k (int): Number of frequencies per input dimension. Must be positive.
        """
        super().__init__()  # Initialize the parent torch.nn.Module
        assert k > 0, "k must be positive"
        self.k: int = k

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Encode input data using Fourier feature mapping.

        Args:
            data (torch.Tensor): Input tensor of shape (batch_size, time_points, data_dims)
                with values in [-1, 1].

        Returns:
            torch.Tensor: Encoded tensor of shape (batch_size, time_points, 2 * k * data_dims).
        """
        batch_size, time_points, data_dims = data.shape

        # Generate k frequencies linearly spaced between 1 and 2^(k-1) and multiply by π.
        frequencies = (
            torch.linspace(1.0, 2.0 ** (self.k - 1), self.k, device=data.device)
            * torch.pi
        )
        # Reshape frequencies to (1, 1, 1, k) for broadcasting over the batch, time, and data dimensions.
        frequencies = frequencies.view(1, 1, 1, self.k)

        # Expand the input tensor to shape (batch_size, time_points, data_dims, 1)
        data_expanded = data.unsqueeze(-1)
        # Multiply input by frequencies: resulting shape is (batch_size, time_points, data_dims, k)
        data_scaled = data_expanded * frequencies

        # Compute sine and cosine features
        sin_features = torch.sin(data_scaled)
        cos_features = torch.cos(data_scaled)

        # Concatenate sine and cosine features along the last dimension,
        # yielding shape (batch_size, time_points, data_dims, 2 * k)
        fourier_features = torch.cat([sin_features, cos_features], dim=-1)

        # Reshape to (batch_size, time_points, 2 * k * data_dims)
        output = fourier_features.view(batch_size, time_points, -1)
        return output

    def compute_output_shape(self, input_size: int) -> int:
        """
        Compute the output size of the Fourier feature encoding given the input size.

        Args:
            input_size (int): The size of the data dimension (n) of the input tensor.

        Returns:
            int: The size of the output dimension, which is 2 * k * input_size.
        """
        return 2 * self.k * input_size


class RBFEncoder(torch.nn.Module):
    """
    Encode input data using a Radial Basis Function (RBF) mapping.

    Given an input tensor of shape (batch_size, time_points, data_dims) with values
    assumed to be normalized (e.g., in [-1, 1]), this encoder computes a set of Gaussian
    RBF activations for each input dimension. Each dimension is encoded into a vector of length
    `num_centers` and the outputs are concatenated, resulting in a final output shape:

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
        Initialize the RBFEncoder.

        Args:
            num_centers (int): Number of RBF centers per input dimension.
            sigma (float): Standard deviation of the Gaussian. If None, it is set to the spacing
                           between centers.
            center_range (Tuple[float, float]): The range (min, max) over which to place the centers.
            sigma_scale (float): A multiplicative factor to scale sigma.
            learn_sigma (bool): If True, sigma is a learnable parameter (learned in log scale).
        """
        super().__init__()
        assert num_centers > 0, "num_centers must be positive"
        self.num_centers = num_centers

        # Compute default sigma if not provided.
        default_sigma = 1.0
        if num_centers > 1:
            default_sigma = (center_range[1] - center_range[0]) / (num_centers - 1)
        sigma_val = sigma if sigma is not None else default_sigma
        sigma_val *= sigma_scale

        # Store sigma in the log-domain.
        log_sigma_val = torch.log(torch.tensor(sigma_val, dtype=torch.float32))
        if learn_sigma:
            self.log_sigma = torch.nn.Parameter(log_sigma_val)
        else:
            self.register_buffer("log_sigma", log_sigma_val)

        # Precompute and register the centers as a buffer.
        centers = torch.linspace(center_range[0], center_range[1], num_centers)
        self.register_buffer("centers", centers)

    @property
    def sigma(self) -> torch.Tensor:
        """
        Dynamically compute sigma by exponentiating the stored log_sigma.
        """
        return torch.exp(self.log_sigma)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Encode input data using Gaussian RBFs.

        Args:
            data (torch.Tensor): Input tensor of shape (batch_size, time_points, data_dims).

        Returns:
            torch.Tensor: RBF encoded tensor of shape
                          (batch_size, time_points, data_dims * num_centers).
        """
        batch_size, time_points, data_dims = data.shape
        # Expand data to shape (batch_size, time_points, data_dims, 1)
        data_expanded = data.unsqueeze(-1)
        # Reshape centers to (1, 1, 1, num_centers) for broadcasting.
        centers = self.centers.view(1, 1, 1, self.num_centers)
        # Compute the squared difference.
        diff = data_expanded - centers
        # Compute the RBF activations: exp( - (x - μ)² / (2σ²) )
        rbf_features = torch.exp(-(diff**2) / (2 * (self.sigma**2)))
        # Flatten the last two dimensions to obtain shape: (batch_size, time_points, data_dims * num_centers)
        output = rbf_features.view(batch_size, time_points, -1)
        return output

    def compute_output_shape(self, input_shape: int) -> int:
        """
        Compute the output shape of the RBF encoding given an input shape.

        Args:
            input_shape (int): The data dim size of the input tensor

        Returns:
            int: The size of the output tensor
        """
        return input_shape * self.num_centers


class LinearEncoder(torch.nn.Module):
    """
    Encode input data using a linear transformation.

    Given an input tensor of shape (batch_size, time_points, data_dims) with values
    assumed to be normalized in the range [-1, 1], this encoder computes two encodings
    for each input element:

        y1 = 0.5 * x + 0.5
        y2 = -0.5 * x + 0.5

    The two outputs are concatenated along the last dimension, yielding an output tensor
    of shape:

        (batch_size, time_points, data_dims * 2)
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute the two linear encodings and concatenate them.

        Args:
            data (torch.Tensor): Input tensor of shape (batch_size, time_points, data_dims).

        Returns:
            torch.Tensor: Encoded tensor of shape (batch_size, time_points, data_dims * 2).
        """
        # Compute the two encodings.
        y1 = 0.5 * data + 0.5
        y2 = -0.5 * data + 0.5
        # Concatenate along the last dimension.
        output = torch.cat([y1, y2], dim=-1)
        return output

    def compute_output_shape(self, input_shape: int) -> int:
        """
        Given the data dimension of the input tensor, compute the output dimension after encoding.

        Args:
            input_shape (int): The data dimension of the input tensor.

        Returns:
            int: The data dimension of the encoded tensor.
        """
        return input_shape * 2


class Linear4DEncoder(torch.nn.Module):
    """
    A linear encoder that maps each input value x (assumed to be in some range, e.g. [-1,1])
    to four outputs via:

        y1 = a*x + b,
        y2 = a*x - b,
        y3 = -a*x + b,
        y4 = -a*x - b.

    Optionally, the four outputs are normalized so that the resulting 4D vector has unit L2 norm.

    Parameters:
      - a (float): scaling factor (default 0.5)
      - b (float): offset (default 0.5)
      - learn_params (bool): if True, a and b are learnable parameters stored in log-scale.
      - normalize (bool): if True, normalize the 4D output for each input element.
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
            # Store parameters in log-domain for stability.
            self.log_a = torch.nn.Parameter(torch.log(torch.tensor(a, dtype=torch.float32)))
            self.log_b = torch.nn.Parameter(torch.log(torch.tensor(b, dtype=torch.float32)))
        else:
            # Precompute and store a and b as buffers (or fixed attributes) to save compute.
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
        """
        Forward pass.

        Args:
          data (torch.Tensor): Input tensor of shape (batch_size, time_points, data_dims)

        Returns:
          torch.Tensor: Encoded tensor of shape (batch_size, time_points, data_dims * 4)
        """
        # Compute the four linear projections.
        y1 = self.a * data + self.b
        y2 = self.a * data - self.b
        y3 = -self.a * data + self.b
        y4 = -self.a * data - self.b

        # Stack along a new last dimension; shape becomes (batch_size, time_points, data_dims, 4)
        out = torch.stack([y1, y2, y3, y4], dim=-1)

        # Optionally normalize the 4D output along the last dimension.
        if self.normalize:
            norm = torch.norm(out, dim=-1, keepdim=True)
            out = out / (norm + 1e-8)

        # Flatten the last two dimensions so that the output shape becomes
        # (batch_size, time_points, data_dims * 4)
        batch_size, time_points, data_dims, _ = out.shape
        out = out.view(batch_size, time_points, data_dims * 4)
        return out

    def compute_output_shape(self, input_shape: int) -> int:
        """
        Given the input feature dimension, returns the output feature dimension.
        """
        return input_shape * 4
