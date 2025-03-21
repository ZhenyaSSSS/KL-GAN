import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence, Independent

# Function for calculating symmetric KL divergence
def compute_mean_std(features, epsilon=1e-10):
    mu = features.mean(dim=0)
    var = features.var(dim=0, unbiased=False) + epsilon
    return mu, var

def symmetric_kl_divergence(real_features, fake_features):
    """
    Computes symmetric KL divergence between real and fake features.
    """
    mu_real, std_real = compute_mean_std(real_features)
    mu_fake, std_fake = compute_mean_std(fake_features)
    real_dist = Independent(Normal(mu_real, std_real), 1)
    fake_dist = Independent(Normal(mu_fake, std_fake), 1)

    kl_real_fake = kl_divergence(real_dist, fake_dist)
    kl_fake_real = kl_divergence(fake_dist, real_dist)
    return 0.5 * (torch.log1p(kl_real_fake) + torch.log1p(kl_fake_real))


class MinibatchDiscrimination(nn.Module):
    """
    A layer to help reduce mode collapse by comparing samples in a batch
    against each other.
    """
    def __init__(self, in_features, out_features, kernel_dims, mean=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_dims = kernel_dims
        self.mean = mean
        self.T = nn.Parameter(torch.Tensor(in_features, out_features, kernel_dims))
        nn.init.normal_(self.T, 0, 1)

    def forward(self, x):
        # x shape: NxA
        # T shape: AxBxC
        matrices = x.mm(self.T.view(self.in_features, -1))
        matrices = matrices.view(-1, self.out_features, self.kernel_dims)
        M = matrices.unsqueeze(0)   # 1xNxBxC
        M_T = M.permute(1, 0, 2, 3) # Nx1xBxC

        norm = torch.abs(M - M_T).sum(3)   # NxNxB
        expnorm = torch.exp(-norm)
        o_b = (expnorm.sum(0) - 1)         # NxB
        if self.mean:
            o_b /= x.size(0) - 1
        x = torch.cat([x, o_b], 1)
        return x


class Discriminator(nn.Module):
    def __init__(self, type_model, minibatch_shader=False, dim=128, use_minibatch=True):
        super(Discriminator, self).__init__()
        self.type_model = type_model
        self.minibatch_shader = minibatch_shader
        self.dim = dim
        self.use_minibatch = use_minibatch

        self.main = nn.Sequential(
            nn.Conv2d(3, dim, 5, stride=2, padding=2),
            nn.LeakyReLU(negative_slope=0.2),
            
            nn.Conv2d(dim, 2 * dim, 5, stride=2, padding=2),
            nn.LeakyReLU(negative_slope=0.2),
            
            nn.Conv2d(2 * dim, 4 * dim, 5, stride=2, padding=2),
            nn.LeakyReLU(negative_slope=0.2),
            
            nn.Conv2d(4 * dim, 8 * dim, 2, stride=2),
            nn.LeakyReLU(negative_slope=0.2),
            
            nn.Conv2d(8 * dim, 16 * dim, 2, stride=2),
            nn.LeakyReLU(negative_slope=0.2),
        )
        
        self.output = nn.Linear(16 * dim, 8)
        if use_minibatch:
            self.minibatch = MinibatchDiscrimination(8, 8, 1)
            self.final = nn.Linear(16, 1)
        else:
            self.final = nn.Linear(8, 1)

    def forward(self, x):
        if self.type_model == "KL-GAN":
            return self.forward_kl(x)
        else:
            return self.forward_average(x)

    def forward_average(self, x):
        out = self.main(x)
        out = out.flatten(1)
        out = self.output(out)
        if self.use_minibatch:
            out = self.minibatch(out)
        out = self.final(out)
        return out

    def forward_kl(self, x):
        """
        For KL-GAN, we chunk real/fake images from x along the batch dimension.
        """
        out = self.main(x)
        out = out.flatten(1)
        out = self.output(out)
        
        if self.use_minibatch:
            if self.minibatch_shader:
                real_features, fake_features = self.minibatch(out).chunk(2, dim=0)
                return symmetric_kl_divergence(real_features, fake_features)
            else:
                real_features, fake_features = out.chunk(2, dim=0)
                return symmetric_kl_divergence(
                    self.minibatch(real_features),
                    self.minibatch(fake_features)
                )
        else:
            real_features, fake_features = out.chunk(2, dim=0)
            return symmetric_kl_divergence(real_features, fake_features)


class StableFlowDiscriminator(nn.Module):
    """
    Multi-scale discriminator that takes a list of images
    of different sizes [4x4, 8x8, ..., resolution x resolution].
    """
    def __init__(
        self,
        resolution=32,
        dim=128,
        type_model="KL-GAN",
        minibatch_shader=False,
        use_minibatch=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.type_model = type_model
        self.minibatch_shader = minibatch_shader
        self.use_minibatch = use_minibatch

        self.num_levels = int(math.log2(self.resolution)) - 2

        # from_rgb blocks + discriminator blocks
        self.from_rgb = nn.ModuleList()
        self.blocks = nn.ModuleList()

        # Based on the generator, at the beginning (4x4) we had dim*4 channels
        in_channels = dim * 4
        prev_channels = 0  # for tracking channels from previous level

        for level in range(self.num_levels):
            out_channels = max(dim, in_channels // 2)

            # from_rgb remains unchanged
            frgb = nn.Conv2d(3, out_channels, kernel_size=1)
            self.from_rgb.append(frgb)

            # Discriminator block now takes out_channels + prev_channels
            block = nn.Sequential(
                nn.Conv2d(out_channels + prev_channels, out_channels, 3, padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.LeakyReLU(0.2),
                nn.AvgPool2d(2)
            )
            self.blocks.append(block)

            prev_channels = out_channels  # save for next level
            in_channels = out_channels

        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.fc = nn.Linear(in_channels * 4 * 4, 8)  # equivalent to output
        if self.use_minibatch:
            self.minibatch = MinibatchDiscrimination(8, 8, 1)
            self.final = nn.Linear(16, 1)
        else:
            self.final = nn.Linear(8, 1)

    def forward(self, multi_res_images):
        """
        multi_res_images is a list of [4x4, 8x8, ..., resolution].
        But the order is usually from smaller to larger.
        In the code below we want to go from larger resolution to smaller -
        so we reverse the list.
        
        If KL-GAN, then in each tensor batch = 2N (concatenated real/fake).
        """
        if self.type_model == "KL-GAN":
            return self.forward_kl(multi_res_images)
        else:
            return self.forward_average(multi_res_images)

    def forward_average(self, x):
        features = self.forward_multiscale(x)  
        if self.use_minibatch:
            features = self.minibatch(features)
        out = self.final(features)
        return out

    def forward_kl(self, x):
        """
        x[i] is [2N, 3, H, W].
        After forward_multiscale we get [2N, 8] (after fc).
        Then chunk into real/fake and compute symmetric_kl_divergence.
        """
        features = self.forward_multiscale(x)
        if self.use_minibatch:
            if self.minibatch_shader:
                real_f, fake_f = self.minibatch(features).chunk(2, dim=0)
                return symmetric_kl_divergence(real_f, fake_f)
            else:
                real_chunk, fake_chunk = features.chunk(2, dim=0)
                return symmetric_kl_divergence(
                    self.minibatch(real_chunk),
                    self.minibatch(fake_chunk)
                )
        else:
            real_chunk, fake_chunk = features.chunk(2, dim=0)
            return symmetric_kl_divergence(real_chunk, fake_chunk)

    def forward_multiscale(self, multi_res_images):
        # Reverse the list to go from larger resolution to smaller
        multi_res_images = multi_res_images[::-1]

        x = None
        for img, frgb, block in zip(multi_res_images, self.from_rgb, self.blocks):
            # from_rgb
            feat = frgb(img)
            if x is None:
                x = feat
            else:
                # concatenate features
                x = torch.cat([x, feat], dim=1)
            # convolutional block + avgpool
            x = block(x)
        x = self.final_conv(x)  # more convolutions at 4x4
        x = x.view(x.size(0), -1)
        x = self.fc(x)          # -> [batch, 8]
        return x                # next minibatch + final 