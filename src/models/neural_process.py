"""
Implements the original Neural Process architecture from Garnelo et al. (2018b)
with optional cross-attention mechanism as in Kim et al. (2019).
with an encoder-decoder structure and latent variable model. This module provides
modular components that can be combined to build Neural Process variants.
"""

# Standard library imports
import warnings
from typing import List, Tuple

# Third-party imports
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Independent, Normal
import torch.nn.functional as F

# Local imports
from .base_np import BaseNP


class MLP(nn.Module):
    """
    A multi-layer perceptron with a configurable architecture.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        """
        Initializes the MLP.

        Args:
            input_dim: Input dimension of the MLP.
            hidden_dims: List of hidden layer dimensions for the MLP.
            output_dim: Output dimension of the MLP.
            activation: Activation function name ("relu", "leaky_relu", "tanh", "sigmoid").
            dropout: Dropout probability for regularization.
        """
        super().__init__()

        layers = []
        prev_dim = input_dim

        # Loop through hidden layers and add them to the layer list
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.LayerNorm(dim))
            layers.append(self._get_activation(activation))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = dim

        # Add the final output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        self.layers = nn.Sequential(*layers)

    def _get_activation(self, name: str) -> nn.Module:
        """Gets an activation function module by its name."""
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "leaky_relu": nn.LeakyReLU(),
        }

        # Warn the user if an unsupported activation is requested
        if name.lower() not in activations:
            available = ", ".join(activations.keys())
            warnings.warn(
                f"Unknown activation '{name}'. Available activations: {available}. "
                f"Falling back to 'relu'.",
                UserWarning,
            )
        return activations.get(name.lower(), nn.ReLU())

    def forward(self, x: Tensor) -> Tensor:
        """Performs a forward pass through the MLP."""
        return self.layers(x)


class NPEncoder(nn.Module):
    """Neural Process encoder that maps (x, y) pairs to high-dim representations."""

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        r_dim: int = 128,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        """
        Initializes the encoder.

        Args:
            x_dim: Input dimension.
            y_dim: Output dimension.
            r_dim: Representation dimension.
            hidden_dims: Hidden layer dimensions.
            activation: Activation function name.
            dropout: Dropout probability for regularization.
        """
        super().__init__()

        self.encoder = MLP(
            input_dim=x_dim + y_dim,
            hidden_dims=hidden_dims,
            output_dim=r_dim,
            activation=activation,
            dropout=dropout,
        )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Encodes (x, y) pairs to high-dim representations.

        Args:
            x: Input features [batch_size, n_points, x_dim].
            y: Output values [batch_size, n_points, y_dim].

        Returns:
            Representations [batch_size, n_points, r_dim].
        """
        # Concatenate inputs along the feature dimension
        inputs = torch.cat([x, y], dim=-1)
        return self.encoder(inputs)


class LatentEncoder(nn.Module):
    """Encodes aggregated high-dim representations to latent distribution parameters."""

    def __init__(
        self,
        r_dim: int = 128,
        z_dim: int = 128,
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        """
        Initializes the latent encoder.

        Args:
            r_dim: Dimension of the aggregated high-dim representation.
            z_dim: Latent dimension.
            activation: Activation function name.
            dropout: Dropout probability for regularization.
        """
        super().__init__()

        # Initialize the encoder
        self.z_dim = z_dim
        self.encoder = MLP(
            input_dim=r_dim,
            hidden_dims=[r_dim],  # one hidden layer
            output_dim=2 * z_dim,  # mean and log_var
            activation=activation,
            dropout=dropout,
        )

    def forward(self, r: Tensor) -> Independent:
        """
        Encodes aggregated representations to a latent distribution.

        Args:
            r: Aggregated representations [batch_size, r_dim].

        Returns:
            The latent distribution q(z|C).
        """
        # Get distribution parameters
        params = self.encoder(r)
        mean, log_var = params.split(self.z_dim, dim=-1)

        # Ensure positive variance for numerical stability
        std = torch.exp(0.5 * log_var).clamp(min=1e-6)

        return Independent(Normal(mean, std), 1)


class NPDecoder(nn.Module):
    """Neural Process decoder that generates predictions from latent samples."""

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        z_dim: int = 128,
        r_dim: int = 128,
        use_attention: bool = False,
        hidden_dims: List[int] = [128, 128, 128, 128],
        activation: str = "relu",
        dropout: float = 0.0,
        min_sigma_e: float = 1e-6,
    ) -> None:
        """
        Initializes the decoder.

        Args:
            x_dim: Input dimension.
            y_dim: Output dimension.
            z_dim: Latent dimension.
            r_dim: Representation dimension.
            use_attention: Whether to use attention in the model.
            hidden_dims: Hidden layer dimensions.
            activation: Activation function name.
            dropout: Dropout probability for regularization.
            min_sigma_e: Minimum global noise scale for numerical stability.
        """
        super().__init__()

        self.y_dim = y_dim
        self.use_attention = use_attention

        # Adjust input dimension based on whether attention is used
        input_dim = x_dim + z_dim
        if use_attention:
            input_dim += r_dim

        self.decoder = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=2 * y_dim,  # mean and log_var
            activation=activation,
            dropout=dropout,
        )
        # Learnable log-scale parameter for expressing global noise variance. 
        self.log_sigma_e = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.min_sigma_e = min_sigma_e

    def forward(self, x: Tensor, z: Tensor, r: Tensor = None) -> Independent:
        """
        Decodes latent samples and target inputs to an output distribution.

        Args:
            x: Target inputs [batch_size, n_target, x_dim].
            z: Latent samples [n_z_samples, batch_size, z_dim].
            r: Target-specific representations [batch_size, n_target, r_dim] (optional)

        Returns:
            The predictive distribution p(y|x, z, r)
        """
        # Expand latent to match the number of target points
        z_expanded = z.unsqueeze(2).expand(-1, -1, x.size(1), -1)
        x_expanded = x.unsqueeze(0).expand(z.size(0), -1, -1, -1)

        # Concatenate and decode
        if self.use_attention:
            if r is None:
                raise ValueError("r must be provided if use_attention is True")
            r_expanded = r.unsqueeze(0).expand(z.size(0), -1, -1, -1)
            inputs = torch.cat([x_expanded, r_expanded, z_expanded], dim=-1)
        else:
            inputs = torch.cat([x_expanded, z_expanded], dim=-1)
            
        outputs = self.decoder(inputs)

        mean, raw_s = outputs.split(self.y_dim, dim=-1) # raw_s is relative to the global noise variance.
        s_i = F.softplus(torch.clamp(raw_s, min=-20, max=20)) # Transform to ensure positivity
        sigma_e = torch.exp(self.log_sigma_e) + self.min_sigma_e
        std_i = sigma_e * torch.sqrt(1 + s_i)

        return Independent(Normal(mean, std_i), 1)


class NeuralProcess(BaseNP):
    """The complete Neural Process model (with optional attention)."""

    def __init__(
        self,
        x_dim,
        y_dim,
        r_dim: int = 128,
        z_dim: int = 128,
        encoder_hidden_dims: List[int] = [256, 256],
        decoder_hidden_dims: List[int] = [128, 128, 128, 128],
        n_z_samples_train: int = 20,
        n_z_samples_test: int = 20,
        activation: str = "relu",
        dropout: float = 0.0,
        use_attention: bool = False,
        num_heads: int = 8,
    ) -> None:
        """
        Initializes the Neural Process model.

        Without attention, only a single latent encoder is used to encode context pairs into representations. 
        With attention, two encoders are used (one for the latent path and one for the deterministic path), and cross-attention is applied to the target inputs.

        See the architecture diagrams in plots/np_architecture/, for a visual overview (inspired by http://yanndubs.github.io/Neural-Process-Family/).

        Args:
            x_dim: Input dimension.
            y_dim: Output dimension.
            r_dim: High-dim representation dimension.
            z_dim: Latent dimension.
            encoder_hidden_dims: Encoder hidden dimensions.
            decoder_hidden_dims: Decoder hidden dimensions.
            n_z_samples_train: Number of latent samples during training.
            n_z_samples_test: Number of latent samples during testing.
            activation: Activation function name.
            dropout: Dropout probability for regularization.
            use_attention: Whether to use cross-attention in the model.
            num_heads: Number of heads for cross-attention.
        """
        super().__init__()
        self.n_z_samples_train = n_z_samples_train
        self.n_z_samples_test = n_z_samples_test
        self.use_attention = use_attention

        # Latent path: encode context pairs, aggregate, and parameterize q(z|C).
        self.encoder = NPEncoder(
            x_dim=x_dim,
            y_dim=y_dim,
            r_dim=r_dim,
            hidden_dims=encoder_hidden_dims,
            activation=activation,
            dropout=dropout,
        )
        self.latent_encoder = LatentEncoder(
            r_dim=r_dim, z_dim=z_dim, activation=activation, dropout=dropout
        )

        # Deterministic path for ANP: use a separate context-pair encoder whose
        # pointwise outputs are attended to by each target input.
        if use_attention:
            self.deterministic_encoder = NPEncoder(
                x_dim=x_dim,
                y_dim=y_dim,
                r_dim=r_dim,
                hidden_dims=encoder_hidden_dims,
                activation=activation,
                dropout=dropout,
            )
            self.query_projection = nn.Linear(x_dim, r_dim)
            self.key_projection = nn.Linear(x_dim, r_dim)
            self.attention = nn.MultiheadAttention(
                embed_dim=r_dim, num_heads=num_heads, batch_first=True
            )

        self.decoder = NPDecoder(
            x_dim=x_dim,
            y_dim=y_dim,
            z_dim=z_dim,
            r_dim=r_dim,
            use_attention=use_attention,
            hidden_dims=decoder_hidden_dims,
            activation=activation,
            dropout=dropout,
        )

    def forward(
        self,
        x_context: Tensor,
        y_context: Tensor,
        x_target: Tensor = None,
    ) -> Tuple[Independent, Independent]:
        """
        Performs a forward pass through the Neural Process.

        Args:
            x_context: Context inputs [batch_size, n_context, x_dim].
            y_context: Context outputs [batch_size, n_context, y_dim].
            x_target: Target inputs [batch_size, n_target, x_dim].

        Returns:
            A tuple of (prediction_distribution, latent_distribution).
        """
        # Latent path: encode context points into representations and aggregate.
        r_latent = self.encoder(x_context, y_context)  # [batch, n_context, r_dim]
        r_agg = r_latent.mean(dim=1)  # Aggregate representations [batch, r_dim]

        # Get the latent distribution q(z|C)
        q_z = self.latent_encoder(r_agg)

        # Sample latent variables z from q(z|C)
        n_samples = self.n_z_samples_train if self.training else self.n_z_samples_test
        z = q_z.rsample([n_samples])  # [n_samples, batch, z_dim]

        p_y = None
        if x_target is not None:
            if self.use_attention:
                # Deterministic path with cross-attention
                r_det = self.deterministic_encoder(
                    x_context, y_context
                )  # [batch, n_context, r_dim]
                # Project x_target and x_context to r_dim for queries and keys
                q = self.query_projection(x_target)  # [batch, n_target, r_dim]
                k = self.key_projection(x_context)   # [batch, n_context, r_dim]
                # Cross attention: query target inputs, key context inputs, value context representations
                r_target, _ = self.attention(q, k, r_det) # [batch, n_target, r_dim]
                # Decode to get the predictive distribution p(y|x, z, r_target)
                p_y = self.decoder(x_target, z, r=r_target)
            else:
                # Decode to get the predictive distribution p(y|x, z)
                p_y = self.decoder(x_target, z)
        return p_y, q_z  # p_y contains mean and std of shape [n_samples, batch_size, n_target, y_dim]
