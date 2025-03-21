import math
import torch
import torch.nn as nn

class Generator(nn.Module):
    """
    KL-GAN Generator with trainable distribution parameters (mu, log_sigma).
    """
    def __init__(self, latent_dim=128, dim=128):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.dim = dim

        # Trainable distribution parameters
        self.mu = nn.Parameter(torch.zeros(latent_dim))
        self.log_sigma = nn.Parameter(torch.zeros(latent_dim))

        # Main architecture
        self.preprocess = nn.Sequential(
            nn.Linear(latent_dim, 4 * 4 * dim),
            nn.Mish(),
            nn.LayerNorm(4 * 4 * dim)
        )
        self.block1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(4 * dim, 2 * dim, 3, 1, 1),
            nn.Mish(),
            nn.ConvTranspose2d(2 * dim, 2 * dim, 5),
            nn.Mish(),
            nn.InstanceNorm2d(2 * dim, affine=True),
            nn.Conv2d(2 * dim, 2 * dim, 3, 1, 1),
            nn.Mish()
        )
        self.block3 = nn.Sequential(
            nn.ConvTranspose2d(2 * dim, 2 * dim, 4, 2, 1),
            nn.Mish(),
            nn.InstanceNorm2d(2 * dim, affine=True),
            nn.Conv2d(2 * dim, 2 * dim, 3, 1, 1),
            nn.Mish()
        )
        self.block4 = nn.Sequential(
            nn.ConvTranspose2d(2 * dim, dim, 4, 2, 1),
            nn.Mish(),
            nn.InstanceNorm2d(dim, affine=True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.Mish()
        )
        self.deconv_out = nn.Conv2d(dim, 3, 3, 1, 1)
        self.tanh = nn.Tanh()

    def forward(self, eps):
        sigma = torch.exp(self.log_sigma)
        z = self.mu + sigma * eps
        
        out = self.preprocess(z)
        out = out.view(-1, 4 * self.dim, 2, 2)
        out = self.block1(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.deconv_out(out)
        out = self.tanh(out)
        return out


class StableFlowGenerator(nn.Module):
    """
    Multi-scale generator with "to_rgb" at each level.
    Generation starts at 4x4, then 8x8, 16x16, etc. up to the specified resolution.
    """
    def __init__(
        self,
        latent_dim=128,
        resolution=32,
        dim=128
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.resolution = resolution
        
        # Calculate number of levels needed (e.g., if resolution=32, levels will be 4->8->16->32)
        self.num_levels = int(math.log2(self.resolution)) - 2  # for 32 -> (log2(32)=5) -> 5 - 2 = 3 levels

        # Distribution parameters (as in KL-GAN Generator)
        self.mu = nn.Parameter(torch.zeros(latent_dim))
        self.log_sigma = nn.Parameter(torch.zeros(latent_dim))

        # Initial block to project latent to 4x4 spatial tensor
        initial_channels = dim * 4
        self.initial = nn.Sequential(
            nn.Linear(latent_dim, 4 * 4 * initial_channels),
            nn.Mish()
        )
        self.initial_norm = nn.LayerNorm([initial_channels, 4, 4])

        # Create blocks for growing resolution
        self.blocks = nn.ModuleList()
        self.to_rgb = nn.ModuleList()
        
        in_channels = initial_channels
        for level in range(self.num_levels):
            out_channels = max(dim, in_channels // 2)
            block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.Mish(),
                nn.InstanceNorm2d(out_channels, affine=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.Mish(),
                nn.InstanceNorm2d(out_channels, affine=True),
            )
            self.blocks.append(block)
            self.to_rgb.append(
                nn.Conv2d(out_channels, 3, kernel_size=1)  # to_rgb for each level
            )
            in_channels = out_channels

        self.tanh = nn.Tanh()

    def forward(self, eps):
        sigma = torch.exp(self.log_sigma)
        z = self.mu + sigma * eps  # sample from parameterized distribution

        # Initial 4x4 tensor
        out = self.initial(z)
        N = out.size(0)
        # Transform to (N, in_channels, 4, 4)
        initial_channels = self.blocks[0][1].in_channels if len(self.blocks) > 0 else self.initial_norm.normalized_shape[0]
        out = out.view(N, initial_channels, 4, 4)
        out = self.initial_norm(out)

        images = []
        current = out
        for block, to_rgb in zip(self.blocks, self.to_rgb):
            current = block(current)
            rgb = to_rgb(current)
            rgb = self.tanh(rgb)
            images.append(rgb)

        # Return the *entire* list: [4x4, 8x8, 16x16, ..., resolution]
        return images 