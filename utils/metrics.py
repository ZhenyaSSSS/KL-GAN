import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Initialize pretrained models for feature extraction
inception_model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
inception_model.cuda().eval()
inception_model.fc = nn.Identity()
for param in inception_model.parameters():
    param.requires_grad = False

RegNet_model = models.regnet_x_3_2gf(weights="RegNet_X_3_2GF_Weights.DEFAULT")
RegNet_model.cuda().eval()
RegNet_model.fc = nn.Identity()
for param in RegNet_model.parameters():
    param.requires_grad = False


def get_features(real_image, fake_image, eps=1e-3):
    """
    Extracts features from real and fake images using Inception v3 and returns
    their mean and covariance.
    """
    real_features = inception_model(real_image)
    fake_features = inception_model(fake_image)
    mu_real, cov_real = real_features.mean(0), torch.cov(real_features.permute(1,0))
    mu_fake, cov_fake = fake_features.mean(0), torch.cov(fake_features.permute(1,0))

    cov_real += eps * torch.eye(cov_real.size(0)).to(cov_real.device)
    cov_fake += eps * torch.eye(cov_fake.size(0)).to(cov_fake.device)

    return mu_real.float(), cov_real.float(), mu_fake.float(), cov_fake.float()


def get_features_RegNet(real_image, fake_image, eps=1e-3):
    """
    Extracts features from real and fake images using RegNet_x_3_2gf and returns
    their mean and covariance.
    """
    real_features = RegNet_model(real_image)
    fake_features = RegNet_model(fake_image)

    mu_real, cov_real = real_features.mean(0), torch.cov(real_features.permute(1,0))
    mu_fake, cov_fake = fake_features.mean(0), torch.cov(fake_features.permute(1,0))

    cov_real += eps * torch.eye(cov_real.size(0)).to(cov_real.device)
    cov_fake += eps * torch.eye(cov_fake.size(0)).to(cov_fake.device)

    return mu_real.float(), cov_real.float(), mu_fake.float(), cov_fake.float()


def calculate_kl_divergence(mu_real, cov_real, mu_fake, cov_fake):
    """
    Symmetric KL divergence between two multivariate Gaussians.
    """
    true_dist = torch.distributions.MultivariateNormal(mu_real.to(torch.float32), cov_real.to(torch.float32))
    fake_dist = torch.distributions.MultivariateNormal(mu_fake.to(torch.float32), cov_fake.to(torch.float32))
    return 0.5 * (
        torch.distributions.kl_divergence(fake_dist, true_dist)
        + torch.distributions.kl_divergence(true_dist, fake_dist)
    )

def calculate_fid(mu_real, cov_real, mu_fake, cov_fake):
    """
    Calculate the FID score between two distributions parameterized by
    (mu_real, cov_real) and (mu_fake, cov_fake).
    """
    mu_real = mu_real.to(torch.float64)
    cov_real = cov_real.to(torch.float64)
    mu_fake = mu_fake.to(torch.float64)
    cov_fake = cov_fake.to(torch.float64)
    
    # (mu1 - mu2)^2
    a = (mu_real - mu_fake).square().sum(dim=-1)
    
    # trace(cov1 + cov2 - 2*sqrt(cov1 cov2))
    b = cov_real.trace() + cov_fake.trace()
    product = cov_real @ cov_fake
    c = 2 * torch.real(torch.linalg.eigvals(product).sqrt().sum())

    return float(a + b - c)


# Functions for multiscale operations
def tensor_to_multiscale(real, max_resolution=32, min_resolution=8):
    """
    Transform tensor [N, 3, max_resolution, max_resolution] into a list
    [4x4, 8x8, 16x16, 32x32].
    """
    images = []
    current_res = min_resolution
    while current_res <= max_resolution:
        scaled = F.interpolate(
            real,
            size=(current_res, current_res),
            mode='bilinear',
            align_corners=False
        )
        images.append(scaled)
        current_res *= 2
    return images

def combine_real_fake_for_kl(real_list, fake_list):
    """
    At each level, concatenate along batch dimension: [N + N, 3, H, W].
    """
    combined = []
    for r, f in zip(real_list, fake_list):
        combined.append(torch.cat([r, f], dim=0))
    return combined


# Functions for scaling latent representations
def scale_latents_to_minus_one_one(x):
    """Scale raw latents -> [-1, 1]."""
    x_scaled = x.div(2 * 3).add(0.5).clamp(0, 1)  # to [0, 1]
    return x_scaled.mul(2).sub(1)  # to [-1, 1]

def unscale_latents_from_minus_one_one(x):
    """Scale [-1, 1] latents -> raw latents."""
    x_zero_one = x.add(1).div(2)      # to [0, 1]
    return x_zero_one.sub(0.5).mul(2 * 3) 