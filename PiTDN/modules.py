"""Neural network modules for the PiTDN-ROM framework.

This module contains the core neural network components:
- Graph autoencoder (Encoder/Decoder) with various convolution options
- SIREN-based layers (SineLayer)
- Tensor Decomposition Network (TDN)
- Vector mapping network (vecmap)
- Invertible MLP for mesh mapping
"""

import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
import numpy as np


class Encoder(torch.nn.Module):
    """Graph encoder with configurable convolution layers.

    Uses graph convolution layers (GMMConv, ChebConv, GCNConv, or GATConv)
    to extract features from graph-structured data, followed by a feed-forward
    network to produce a latent representation.

    Args:
        hidden_channels: List of hidden channel sizes for each convolution layer.
        bottleneck: Size of the bottleneck (latent) layer.
        input_size: Number of input node features.
        ffn: Size of the intermediate feed-forward layer.
        skip: If True, concatenate input features with each layer's output.
        act: Activation function (default: F.elu).
        conv: Convolution type, one of ['GMMConv', 'ChebConv', 'GCNConv', 'GATConv'].
    """

    def __init__(self, hidden_channels, bottleneck, input_size, ffn, skip,
                 act=F.elu, conv='GMMConv'):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.depth = len(self.hidden_channels)
        self.act = act
        self.ffn = ffn
        self.skip = skip
        self.bottleneck = bottleneck
        self.input_size = input_size
        self.conv = conv

        self.down_convs = torch.nn.ModuleList()
        for i in range(self.depth - 1):
            if self.conv == 'GMMConv':
                self.down_convs.append(
                    gnn.GMMConv(self.hidden_channels[i],
                                self.hidden_channels[i + 1], dim=1,
                                kernel_size=5))
            elif self.conv == 'ChebConv':
                self.down_convs.append(
                    gnn.ChebConv(self.hidden_channels[i],
                                 self.hidden_channels[i + 1], K=5))
            elif self.conv == 'GCNConv':
                self.down_convs.append(
                    gnn.GCNConv(self.hidden_channels[i],
                                self.hidden_channels[i + 1]))
            elif self.conv == 'GATConv':
                self.down_convs.append(
                    gnn.GATConv(self.hidden_channels[i],
                                self.hidden_channels[i + 1]))
            else:
                raise NotImplementedError(
                    'Invalid convolution selected. '
                    'Please select one of [GMMConv, ChebConv, GCNConv, GATConv]')

        self.fc_in1 = nn.Linear(self.input_size * self.hidden_channels[-1],
                                self.ffn)
        self.fc_in2 = nn.Linear(self.ffn, self.bottleneck)
        self.reset_parameters()

    def encoder(self, data):
        """Encode graph data into latent representation."""
        x = data.x
        for layer in self.down_convs:
            if self.conv in ['GMMConv', 'ChebConv', 'GCNConv']:
                x = self.act(layer(x, data.edge_index, data.edge_weight))
            elif self.conv == 'GATConv':
                x = self.act(layer(x, data.edge_index, data.edge_attr))
            if self.skip:
                x = x + data.x

        x = x.reshape(data.num_graphs,
                      self.input_size * self.hidden_channels[-1])
        x = self.act(self.fc_in1(x))
        x = self.fc_in2(x)
        return x

    def reset_parameters(self):
        """Reset all parameters with Kaiming uniform initialization."""
        for conv in self.down_convs:
            conv.reset_parameters()
            for name, param in conv.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                else:
                    nn.init.kaiming_uniform_(param)

    def forward(self, data):
        return self.encoder(data)


class Decoder(torch.nn.Module):
    """Graph decoder with configurable convolution layers.

    Reconstructs graph node features from a latent representation using
    graph convolution up-sampling layers.

    Args:
        hidden_channels: List of hidden channel sizes (reversed from encoder).
        bottleneck: Size of the bottleneck (latent) layer.
        input_size: Number of output node features.
        ffn: Size of the intermediate feed-forward layer.
        skip: If True, add skip connections from input features.
        act: Activation function (default: F.elu).
        conv: Convolution type, one of ['GMMConv', 'ChebConv', 'GCNConv', 'GATConv'].
    """

    def __init__(self, hidden_channels, bottleneck, input_size, ffn, skip,
                 act=F.elu, conv='GMMConv'):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.depth = len(self.hidden_channels)
        self.act = act
        self.ffn = ffn
        self.skip = skip
        self.bottleneck = bottleneck
        self.input_size = input_size
        self.conv = conv

        self.fc_out1 = nn.Linear(self.bottleneck, self.ffn)
        self.fc_out2 = nn.Linear(self.ffn,
                                 self.input_size * self.hidden_channels[-1])

        self.up_convs = torch.nn.ModuleList()
        for i in range(self.depth - 1):
            if self.conv == 'GMMConv':
                self.up_convs.append(
                    gnn.GMMConv(self.hidden_channels[self.depth - i - 1],
                                self.hidden_channels[self.depth - i - 2],
                                dim=1, kernel_size=5))
            elif self.conv == 'ChebConv':
                self.up_convs.append(
                    gnn.ChebConv(self.hidden_channels[self.depth - i - 1],
                                 self.hidden_channels[self.depth - i - 2],
                                 K=5))
            elif self.conv == 'GCNConv':
                self.up_convs.append(
                    gnn.GCNConv(self.hidden_channels[self.depth - i - 1],
                                self.hidden_channels[self.depth - i - 2]))
            elif self.conv == 'GATConv':
                self.up_convs.append(
                    gnn.GATConv(self.hidden_channels[self.depth - i - 1],
                                self.hidden_channels[self.depth - i - 2]))
            else:
                raise NotImplementedError(
                    'Invalid convolution selected. '
                    'Please select one of [GMMConv, ChebConv, GCNConv, GATConv]')

        self.reset_parameters()

    def decoder(self, x, data):
        """Decode latent representation back to graph node features."""
        x = self.act(self.fc_out1(x))
        x = self.act(self.fc_out2(x))
        h = x.reshape(data.num_graphs * self.input_size,
                      self.hidden_channels[-1])
        x = h
        idx = 0
        for layer in self.up_convs:
            if self.conv in ['GMMConv', 'ChebConv', 'GCNConv']:
                x = layer(x, data.edge_index, data.edge_weight)
            elif self.conv == 'GATConv':
                x = layer(x, data.edge_index, data.edge_attr)
            if idx != self.depth - 2:
                x = self.act(x)
            if self.skip:
                x = x + h
            idx += 1
        return x

    def reset_parameters(self):
        """Reset all parameters with Kaiming uniform initialization."""
        for conv in self.up_convs:
            conv.reset_parameters()
            for name, param in conv.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                else:
                    nn.init.kaiming_uniform_(param)

    def forward(self, x, data):
        return self.decoder(x, data)


class SineLayer(nn.Module):
    """SIREN-style sinusoidal activation layer.

    Implements a linear layer followed by sin(omega_0 * x) activation,
    with specialized weight initialization for SIREN networks.

    Args:
        in_features: Input feature dimension.
        out_features: Output feature dimension.
        bias: Whether to include bias.
        is_first: Whether this is the first layer (uses different init).
        omega_0: Frequency scaling factor (default: 2).
    """

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=2):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        """Initialize weights following the SIREN initialization scheme."""
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(
                    -1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class vecmap(nn.Module):
    """Vector mapping network for parameter-to-mode mapping.

    Maps a latent vector to three factor matrices (U, V, W) for the
    Tucker decomposition used in the Tensor Decomposition Network.

    Args:
        n_1, n_2, n_3: Output dimensions for U, V, W factor matrices.
        infeature: Input feature dimension (from the autoencoder bottleneck).
        hidden_dim: Hidden dimension for the MLP layers (default: 2048).
        num_layers: Number of layers in each MLP (default: 1).
    """

    def __init__(self, n_1, n_2, n_3, infeature, hidden_dim=2048,
                 num_layers=1):
        super(vecmap, self).__init__()
        self.n_1 = n_1
        self.n_2 = n_2
        self.n_3 = n_3
        self.infeature = infeature

        def build_mlp(input_dim, output_dim):
            layers = []
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(nn.ReLU())
                input_dim = hidden_dim
            layers.append(nn.Linear(input_dim, output_dim))
            return nn.Sequential(*layers)

        self.U_input = build_mlp(infeature, n_1)
        self.V_input = build_mlp(infeature, n_2)
        self.W_input = build_mlp(infeature, n_3)

    def forward(self, x):
        """Map latent vector x to factor matrices U, V, W.

        Args:
            x: Input latent vector of shape (infeature,).

        Returns:
            Tuple of (U_i, V_i, W_i) where each is a column vector
            of shape (n_k, 1).
        """
        x_reshape = x.view(1, -1)
        U_i = self.U_input(x_reshape).T
        V_i = self.V_input(x_reshape).T
        W_i = self.W_input(x_reshape).T
        return U_i, V_i, W_i


class TDN(nn.Module):
    """Tensor Decomposition Network for structured field reconstruction.

    Reconstructs a 3D tensor (n_1 x n_2 x n_3) from a core tensor and
    1D coordinate inputs using learned factor matrices via SIREN networks.

    Args:
        r_1, r_2, r_3: Ranks of the Tucker decomposition.
        mid_channel: Hidden dimension for the SIREN networks.
    """

    def __init__(self, r_1, r_2, r_3, mid_channel):
        super(TDN, self).__init__()
        self.r_1 = r_1
        self.r_2 = r_2
        self.r_3 = r_3
        self.mid_channel = mid_channel

        self.U_net = nn.Sequential(
            SineLayer(1, mid_channel, is_first=True),
            SineLayer(mid_channel, mid_channel, is_first=True),
            nn.Linear(mid_channel, r_1))

        self.V_net = nn.Sequential(
            SineLayer(1, mid_channel, is_first=True),
            SineLayer(mid_channel, mid_channel, is_first=True),
            nn.Linear(mid_channel, r_2))

        self.W_net = nn.Sequential(
            SineLayer(1, mid_channel, is_first=True),
            SineLayer(mid_channel, mid_channel, is_first=True),
            nn.Linear(mid_channel, r_3))

    def forward(self, centre, U_i, V_i, W_i):
        """Reconstruct the full tensor from the core and coordinate inputs.

        Args:
            centre: Core tensor of shape (r_1, r_2, r_3).
            U_i: 1D coordinate input for U network, shape (n_1, 1).
            V_i: 1D coordinate input for V network, shape (n_2, 1).
            W_i: 1D coordinate input for W network, shape (n_3, 1).

        Returns:
            Tuple of (reconstructed_tensor, U, V, W, core).
        """
        U = self.U_net(U_i)
        V = self.V_net(V_i)
        W = self.W_net(W_i)
        device = U_i.device
        centre = centre.to(device)
        core = centre
        centre = centre.permute(1, 2, 0)
        centre = centre @ U.t()
        centre = centre.permute(2, 1, 0)
        centre = centre @ V.t()
        centre = centre.permute(0, 2, 1)
        centre = centre @ W.t()
        return centre, U, V, W, core


class ForwardMapping(nn.Module):
    """Forward mapping network from input to output dimension."""

    def __init__(self, input_dim, output_dim, hidden_dim):
        super(ForwardMapping, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        return self.net(x)


class InverseMapping(nn.Module):
    """Inverse mapping network from output back to input dimension."""

    def __init__(self, output_dim, input_dim, hidden_dim):
        super(InverseMapping, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim))

    def forward(self, y):
        return self.net(y)


class InvertibleMLP(nn.Module):
    """Invertible model using forward and inverse mapping networks.

    Args:
        input_dim: Dimension of the input space.
        output_dim: Dimension of the output space.
        hidden_dim: Hidden layer dimension.
    """

    def __init__(self, input_dim, output_dim, hidden_dim):
        super(InvertibleMLP, self).__init__()
        self.forward_net = ForwardMapping(input_dim, output_dim, hidden_dim)
        self.inverse_net = InverseMapping(output_dim, input_dim, hidden_dim)

    def forward(self, x):
        return self.forward_net(x)

    def inverse(self, y):
        return self.inverse_net(y)
