"""
Architecture builders for ePC experiments.

Provides ResNet-18 (CIFAR-10) and MLP (MNIST) architectures compatible
with the PCE and PCESkipConnection models.

Skip connection modules (SaveIdentity, AddIdentity, LayerWithResidual)
implement the tuple-based (activity, identity) protocol for residual
connections in the ePC framework.

Reference: https://github.com/cgoemaere/error_based_PC (cifar branch)
"""

import torch.nn as nn


class SaveIdentity(nn.Module):
    """Save current activity as identity for skip connection.

    At the start of a residual block, captures the current activity
    to be added back at the end (via AddIdentity). Optionally applies
    a downsample projection when dimensions change.
    """

    def __init__(self, identity_downsample=None):
        super().__init__()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        if isinstance(x, tuple):
            if self.identity_downsample is not None:
                x_new = self.identity_downsample(x[0])
            else:
                x_new = x[0]
            x = (x[0], x_new)
        return x


class AddIdentity(nn.Module):
    """Add saved identity to activity (residual connection).

    At the end of a residual block, adds the saved identity to the
    current activity: output = F(x) + x.
    """

    def forward(self, x):
        if isinstance(x, tuple):
            x = (x[0] + x[1], 0.0)
        return x


class LayerWithResidual(nn.Module):
    """Wrap a layer to process only activity, passing identity through.

    This adapter allows standard nn.Module layers to work within the
    (activity, identity) tuple protocol used by skip connections.
    """

    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        if isinstance(x, tuple):
            return (self.layer(x[0]), x[1])
        return self.layer(x)


def get_resnet_block(in_channels, out_channels, stride=1, avg_pool=False):
    """Build a ResNet basic block, split into 2 ePC layers.

    Each block has 2 conv layers with a skip connection. For ePC, this is
    split into block1 (SaveIdentity + conv1 + bn + relu) and block2
    (conv2 + bn + AddIdentity + relu). Errors are placed between blocks.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        stride: Stride for first conv (2 for downsampling).
        avg_pool: Whether to append average pooling (for last stage).

    Returns:
        (block1, block2): Two nn.Sequential modules.
    """
    conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
    bn1 = nn.BatchNorm2d(out_channels)
    conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
    bn2 = nn.BatchNorm2d(out_channels)
    relu = nn.ReLU()

    identity_downsample = None
    if stride > 1:
        identity_downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
        )

    block1 = nn.Sequential(
        SaveIdentity(identity_downsample),
        LayerWithResidual(conv1),
        LayerWithResidual(bn1),
        LayerWithResidual(relu),
    )

    block2_layers = [
        LayerWithResidual(conv2),
        LayerWithResidual(bn2),
        AddIdentity(),
        LayerWithResidual(relu),
    ]
    if avg_pool:
        block2_layers.append(LayerWithResidual(nn.AvgPool2d(4, 4)))

    block2 = nn.Sequential(*block2_layers)
    return block1, block2


def get_resnet18_cifar10(num_classes=10):
    """ResNet-18 architecture for CIFAR-10.

    Matches the ePC paper's ResNet-18 (Goemaere et al. 2025):
    - 4 stages with 1 block each (8 conv layers + initial + FC = 10 layers)
    - Each block split into 2 ePC layers for error placement
    - Total: 10 ePC layers, 8 error terms
    - Target accuracy: 92.17%

    Returns:
        List of nn.Module layers for PCESkipConnection.
    """
    return [
        LayerWithResidual(nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )),
        *get_resnet_block(64, 64, stride=1),
        *get_resnet_block(64, 128, stride=2),
        *get_resnet_block(128, 256, stride=2),
        *get_resnet_block(256, 512, stride=2, avg_pool=True),
        LayerWithResidual(nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, num_classes),
        )),
    ]


def get_mlp_mnist(hidden_size=128, num_hidden=3):
    """MLP architecture for MNIST validation.

    Same structure as eBPC baseline: [784, 128, 128, 128, 10].
    Uses standard PCE (no skip connections needed).

    Args:
        hidden_size: Hidden layer size.
        num_hidden: Number of hidden layers.

    Returns:
        List of nn.Sequential layers for PCE.
    """
    layers = [nn.Sequential(nn.Linear(784, hidden_size), nn.ReLU())]
    for _ in range(num_hidden - 1):
        layers.append(nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU()))
    layers.append(nn.Sequential(nn.Linear(hidden_size, 10)))
    return layers
