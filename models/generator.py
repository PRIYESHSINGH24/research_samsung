import torch
import torch.nn as nn


class UAPGenerator(nn.Module):
    """
    Generates UAPs from random noise
    Paper: 4 deconv layers, tanh output scaled by epsilon
    """
    def __init__(self, latent_dim=10, channels=1, epsilon=10/255):
        super(UAPGenerator, self).__init__()
        self.epsilon = epsilon
        
        self.deconv = nn.Sequential(
            # latent_dim -> 4x4
            nn.ConvTranspose2d(latent_dim, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 4x4 -> 8x8
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 8x8 -> 16x16
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 16x16 -> 32x32
            nn.ConvTranspose2d(64, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, z):
        # z shape: [B, latent_dim, 1, 1]
        if z.dim() == 2:
            z = z.unsqueeze(-1).unsqueeze(-1)
        uap = self.deconv(z)
        # Scale to epsilon ball
        uap = uap * self.epsilon
        return uap


if __name__ == "__main__":
    gen = UAPGenerator(latent_dim=10, channels=1)
    z = torch.randn(4, 10)
    uap = gen(z)
    print(f"Generator Output: {uap.shape}")
    print(f"UAP Range: [{uap.min():.4f}, {uap.max():.4f}]")
    print(f"Generator Params: {sum(p.numel() for p in gen.parameters()):,}")