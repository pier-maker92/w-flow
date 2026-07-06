import torch
import torch.nn as nn
from ..configs import VelocityAEConfig

class VelocityAutoencoder(nn.Module):
    """
    A separate Autoencoder for compressing and reconstructing the velocity field.
    Architecture:
    Encoder: [B, C, H, W] -> [B, bottleneck_dim]
    Decoder: [B, bottleneck_dim] -> [B, C, H, W]
    """
    def __init__(self, config: VelocityAEConfig, in_channels: int = 1, image_size: int = 28):
        super().__init__()
        self.config = config
        self.in_channels = in_channels
        self.image_size = image_size
        
        # Simple Convolutional Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, config.hidden_dim, kernel_size=4, stride=2, padding=1), # 14x14
            nn.ReLU(),
            nn.Conv2d(config.hidden_dim, config.hidden_dim * 2, kernel_size=4, stride=2, padding=1), # 7x7
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(config.hidden_dim * 2 * 7 * 7, config.bottleneck_dim)
        )
        
        # Simple Convolutional Decoder
        self.decoder = nn.Sequential(
            nn.Linear(config.bottleneck_dim, config.hidden_dim * 2 * 7 * 7),
            nn.Unflatten(1, (config.hidden_dim * 2, 7, 7)),
            nn.ReLU(),
            nn.ConvTranspose2d(config.hidden_dim * 2, config.hidden_dim, kernel_size=4, stride=2, padding=1), # 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(config.hidden_dim, in_channels, kernel_size=4, stride=2, padding=1) # 28x28
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
