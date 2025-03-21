import gc
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision
import wandb
from torchvision import transforms

# Import our modules
from paper.models.generators import Generator, StableFlowGenerator
from paper.models.discriminators import Discriminator, StableFlowDiscriminator
from paper.utils.metrics import get_features, get_features_RegNet, calculate_fid, calculate_kl_divergence, combine_real_fake_for_kl, tensor_to_multiscale, inception_model


class GAN_Training(pl.LightningModule):
    """
    LightningModule for training KL-GAN (and other variants).
    Added hyperparameter use_multiscale to enable multi-scale mode.
    """
    def __init__(
        self,
        learning_rate: float = 0.00002,
        batch_size: int = 256,
        seed_value: int = 1,
        type_model: str = "KL-GAN",
        latent_dim: int = 128,
        dim: int = 128,
        use_minibatch: bool = True,
        # New hyperparameter
        use_multiscale: bool = False,
        resolution: int = 32,   # to pass to multi-scale G/D
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        
        if seed_value is not None:
            pl.seed_everything(seed_value)
            torch.manual_seed(seed_value)
            torch.cuda.manual_seed(seed_value)
            random.seed(seed_value)

        # If multi-scale mode is enabled
        if self.hparams.use_multiscale:
            # Multi-scale G/D
            self.generator = StableFlowGenerator(
                latent_dim=latent_dim, 
                resolution=resolution,
                dim=dim
            )
            self.discriminator = StableFlowDiscriminator(
                resolution=resolution,
                dim=dim,
                type_model=type_model,
                use_minibatch=use_minibatch
            )
        else:
            # Standard implementations
            self.generator = Generator(latent_dim=latent_dim, dim=dim)
            self.discriminator = Discriminator(
                type_model=type_model, 
                dim=dim,
                use_minibatch=use_minibatch
            )
        

    def compute_gradient_penalty(self, real_data, fake_data):
        """
        Gradient penalty for WGAN-GP.
        """
        alpha = torch.rand(real_data.size(0), 1, 1, 1, device=self.device)
        alpha = alpha.expand(real_data.size())

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = self.discriminator(interpolates)
        grad_outputs = torch.ones(disc_interpolates.size(), device=self.device)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def r1_penalty(self, real_data):
        """
        R1 penalty for real images (StyleGAN approach).
        """
        real_data.requires_grad = True
        real_pred = self.discriminator(real_data)
        grad_real = torch.autograd.grad(
            outputs=real_pred.sum(),
            inputs=real_data,
            create_graph=True
        )[0]
        r1_reg = torch.mean(grad_real.pow(2).sum(dim=[i for i in range(1, grad_real.ndim)]))
        return r1_reg

    def on_validation_epoch_end(self):
        gc.collect()
        # Save checkpoint every 50 epochs
        if self.current_epoch % 50 == 0:
            self.trainer.save_checkpoint(filepath=f"./checkpoint_{self.hparams.type_model}_seed{self.hparams.seed_value}.ckpt")

    def diversity_loss(self, fake_images):
        """
        A simple diversity measure: average pairwise L2 distances among samples.
        Minimizing the negative encourages more variety.
        """
        batch_size = fake_images.size(0)
        fake_images_flat = fake_images.view(batch_size, -1)
        distances = torch.pdist(fake_images_flat, p=2)
        diversity = distances.mean()
        return -diversity
        
    def training_step(self, batch, batch_idx):
        """
        Manual training loop step that alternates between G and D updates 
        based on the chosen adversarial objective.
        """
        true = batch  # [N, 3, H, W]
        noise = torch.randn((true.shape[0], self.hparams.latent_dim), device=self.device)

        optimizer_dis = self.optimizers()[1]
        optimizer_gen = self.optimizers()[0]
        optimizer_dis.zero_grad()
        optimizer_gen.zero_grad()

        # If multiscale mode, prepare real_list, fake_list and follow the logic
        if self.hparams.use_multiscale:
            real_list = tensor_to_multiscale(true, max_resolution=self.hparams.resolution, min_resolution=8)
            fake_list = self.generator(noise)  # [8x8, ..., resolution]
            if batch_idx % 8 == 0:
                self.log_data(real_list[-1], fake_list[-1])
            # Choose loss function
            if self.hparams.type_model == "KL-GAN":
                # Combine real+fake
                combined_list = combine_real_fake_for_kl(real_list, fake_list)
                kl = self.discriminator(combined_list)  # forward_kl
                self.log('Fake_dist/Train', kl, prog_bar=True, on_epoch=True)

                # Generator: backward with -KL
                (-kl).backward()
                for name, param in self.generator.named_parameters():
                    if param.grad is not None:
                        param.grad.data = -param.grad.data
                optimizer_gen.step()
                optimizer_dis.step()
                return fake_list[-1].detach()  # return the largest level
                
            elif self.hparams.type_model == "LS-GAN":
                return self.ls_gan_step_multiscale(optimizer_gen, optimizer_dis, noise, real_list, fake_list, batch_idx)
            elif self.hparams.type_model == "WGAN-GP":
                return self.wgan_gp_step_multiscale(optimizer_gen, optimizer_dis, noise, real_list, fake_list, batch_idx)
            elif self.hparams.type_model == "Hinge-GAN":
                return self.hinge_gan_step_multiscale(optimizer_gen, optimizer_dis, noise, real_list, fake_list, batch_idx)
            elif self.hparams.type_model == "R1-GAN":
                return self.r1_regularized_hinge_step_multiscale(optimizer_gen, optimizer_dis, noise, real_list, fake_list, batch_idx)

        else:
            # If NOT multiscale mode, use old logic:
            if self.hparams.type_model == "KL-GAN":
                fake = self.kl_gan_step(optimizer_gen, optimizer_dis, noise, true, batch_idx)
            elif self.hparams.type_model == "LS-GAN":
                fake = self.ls_gan_step(optimizer_gen, optimizer_dis, noise, true, batch_idx)
            elif self.hparams.type_model == "WGAN-GP":
                fake = self.wgan_gp_step(optimizer_gen, optimizer_dis, noise, true, batch_idx)
            elif self.hparams.type_model == "Hinge-GAN":
                fake = self.hinge_gan_step(optimizer_gen, optimizer_dis, noise, true, batch_idx)
            elif self.hparams.type_model == "R1-GAN":
                fake = self.r1_regularized_hinge_step(optimizer_gen, optimizer_dis, noise, true, batch_idx)

            return fake

    def log_data(self, true, fake):
        with torch.no_grad():
            mu_real, cov_real, mu_fake, cov_fake = get_features_RegNet(true, fake, eps=10)
            kl_div = calculate_kl_divergence(mu_real, cov_real, mu_fake, cov_fake)
            self.log('KL_divergence/Train', kl_div, prog_bar=True, on_epoch=True)

            dv = self.diversity_loss(fake)
            self.log('Diversity/Train', -dv, prog_bar=True, on_epoch=True)

    # Different adversarial objectives (original)
    def ls_gan_step(self, optimizer_gen, optimizer_dis, noise, true, batch_idx):
        # Generator step
        with optimizer_gen.toggle_model():
            fake = self.generator(noise)
            if batch_idx % 8 == 0:
                self.log_data(true, fake)
            g_loss = self.discriminator(fake)
            self.log('g_loss_ls_gan/Train', g_loss.mean(), prog_bar=True, on_epoch=True)
            torch.mean(g_loss ** 2).backward()
            optimizer_gen.step()

        # Discriminator step
        fake = fake.detach()
        with optimizer_dis.toggle_model():
            true_loss = self.discriminator(true)
            fake_loss = self.discriminator(fake)
            self.log('true_loss_ls_gan/Train', true_loss.mean(), prog_bar=True, on_epoch=True)
            self.log('fake_loss_ls_gan/Train', fake_loss.mean(), prog_bar=True, on_epoch=True)

            true_loss = torch.mean((true_loss) ** 2)
            fake_loss = torch.mean((fake_loss - 1) ** 2)
            loss = (true_loss + fake_loss)
            loss.backward()
            optimizer_dis.step()

        return fake

    def kl_gan_step(self, optimizer_gen, optimizer_dis, noise, true, batch_idx):
        fake = self.generator(noise)
        if batch_idx % 8 == 0:
            self.log_data(true, fake)
        kl = self.discriminator(torch.cat([true, fake], dim=0))
        self.log('Fake_dist/Train', kl, prog_bar=True, on_epoch=True)

        (-kl).backward()
        for name, param in self.generator.named_parameters():
            if param.grad is not None:
                param.grad.data = -param.grad.data

        optimizer_gen.step()
        optimizer_dis.step()
        return fake.detach()

    def wgan_gp_step(self, optimizer_gen, optimizer_dis, noise, true, batch_idx):
        with optimizer_gen.toggle_model():
            fake = self.generator(noise)
            if batch_idx % 8 == 0:
                self.log_data(true, fake)
            gen_loss = -self.discriminator(fake).mean()
            gen_loss.backward()
            optimizer_gen.step()

        fake = fake.detach()
        with optimizer_dis.toggle_model():
            real_pred = self.discriminator(true)
            fake_pred = self.discriminator(fake)
            gp = self.compute_gradient_penalty(true, fake)
            dis_loss = (fake_pred.mean() - real_pred.mean()) + 10.0 * gp
            dis_loss.backward()
            optimizer_dis.step()

        self.log('g_loss_wgan_gp/Train', gen_loss, prog_bar=True, on_epoch=True)
        self.log('d_loss_wgan_gp/Train', dis_loss, prog_bar=True, on_epoch=True)
        return fake

    def hinge_gan_step(self, optimizer_gen, optimizer_dis, noise, true, batch_idx):
        with optimizer_gen.toggle_model():
            fake = self.generator(noise)
            if batch_idx % 8 == 0:
                self.log_data(true, fake)
            gen_loss = -self.discriminator(fake).mean()
            gen_loss.backward()
            optimizer_gen.step()

        fake = fake.detach()
        with optimizer_dis.toggle_model():
            real_pred = self.discriminator(true)
            fake_pred = self.discriminator(fake)
            d_loss_real = torch.mean(F.relu(1 - real_pred))
            d_loss_fake = torch.mean(F.relu(1 + fake_pred))
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_dis.step()

        self.log('g_loss_hinge/Train', gen_loss, prog_bar=True, on_epoch=True)
        self.log('d_loss_hinge/Train', d_loss, prog_bar=True, on_epoch=True)
        return fake

    def r1_regularized_hinge_step(self, optimizer_gen, optimizer_dis, noise, true, batch_idx):
        with optimizer_gen.toggle_model():
            fake = self.generator(noise)
            if batch_idx % 8 == 0:
                self.log_data(true, fake)
            gen_loss = -self.discriminator(fake).mean()
            gen_loss.backward()
            optimizer_gen.step()

        fake = fake.detach()
        with optimizer_dis.toggle_model():
            real_pred = self.discriminator(true)
            fake_pred = self.discriminator(fake)
            d_loss_real = torch.mean(F.relu(1 - real_pred))
            d_loss_fake = torch.mean(F.relu(1 + fake_pred))
            d_loss = d_loss_real + d_loss_fake
            r1 = self.r1_penalty(true) * 10.0
            total_loss = d_loss + r1
            total_loss.backward()
            optimizer_dis.step()

        self.log('g_loss_r1_hinge/Train', gen_loss, prog_bar=True, on_epoch=True)
        self.log('d_loss_r1_hinge/Train', d_loss, prog_bar=True, on_epoch=True)
        self.log('r1_penalty/Train', r1, prog_bar=True, on_epoch=True)
        return fake

    # New multiscale methods (following similar logic at the "last" level)
    def ls_gan_step_multiscale(self, optimizer_gen, optimizer_dis, noise, real_list, fake_list, batch_idx):
        with optimizer_gen.toggle_model():
            fake_high = fake_list[-1]
            true_high = real_list[-1]
            if batch_idx % 8 == 0:
                self.log_data(true_high, fake_high)
            g_loss = self.discriminator(fake_list)
            torch.mean(g_loss ** 2).backward()
            optimizer_gen.step()

        fake_high = fake_high.detach()
        with optimizer_dis.toggle_model():
            true_loss = self.discriminator(real_list)
            fake_loss = self.discriminator(fake_list)
            true_loss = torch.mean((true_loss) ** 2)
            fake_loss = torch.mean((fake_loss - 1) ** 2)
            loss = (true_loss + fake_loss)
            loss.backward()
            optimizer_dis.step()

        return fake_high

    def wgan_gp_step_multiscale(self, optimizer_gen, optimizer_dis, noise, real_list, fake_list, batch_idx):
        with optimizer_gen.toggle_model():
            fake_high = fake_list[-1]
            true_high = real_list[-1]
            if batch_idx % 8 == 0:
                self.log_data(true_high, fake_high)
            gen_loss = -self.discriminator(fake_list).mean()
            gen_loss.backward()
            optimizer_gen.step()

        fake_high = fake_high.detach()
        with optimizer_dis.toggle_model():
            real_pred = self.discriminator(real_list)
            fake_pred = self.discriminator(fake_list)
            # GP at the last level:
            gp = self.compute_gradient_penalty(true_high, fake_high)
            dis_loss = (fake_pred.mean() - real_pred.mean()) + 10.0 * gp
            dis_loss.backward()
            optimizer_dis.step()

        self.log('g_loss_wgan_gp/Train', gen_loss, prog_bar=True, on_epoch=True)
        self.log('d_loss_wgan_gp/Train', dis_loss, prog_bar=True, on_epoch=True)
        return fake_high

    def hinge_gan_step_multiscale(self, optimizer_gen, optimizer_dis, noise, real_list, fake_list, batch_idx):
        with optimizer_gen.toggle_model():
            fake_high = fake_list[-1]
            true_high = real_list[-1]
            if batch_idx % 8 == 0:
                self.log_data(true_high, fake_high)
            gen_loss = -self.discriminator(fake_list).mean()
            gen_loss.backward()
            optimizer_gen.step()

        fake_high = fake_high.detach()
        with optimizer_dis.toggle_model():
            real_pred = self.discriminator(real_list)
            fake_pred = self.discriminator(fake_list)
            d_loss_real = torch.mean(F.relu(1 - real_pred))
            d_loss_fake = torch.mean(F.relu(1 + fake_pred))
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_dis.step()

        self.log('g_loss_hinge/Train', gen_loss, prog_bar=True, on_epoch=True)
        self.log('d_loss_hinge/Train', d_loss, prog_bar=True, on_epoch=True)
        return fake_high

    def r1_regularized_hinge_step_multiscale(self, optimizer_gen, optimizer_dis, noise, real_list, fake_list, batch_idx):
        with optimizer_gen.toggle_model():
            fake_high = fake_list[-1]
            true_high = real_list[-1]
            if batch_idx % 8 == 0:
                self.log_data(true_high, fake_high)
            gen_loss = -self.discriminator(fake_list).mean()
            gen_loss.backward()
            optimizer_gen.step()

        fake_high = fake_high.detach()
        with optimizer_dis.toggle_model():
            real_pred = self.discriminator(real_list)
            fake_pred = self.discriminator(fake_list)
            d_loss_real = torch.mean(F.relu(1 - real_pred))
            d_loss_fake = torch.mean(F.relu(1 + fake_pred))
            d_loss = d_loss_real + d_loss_fake
            r1 = self.r1_penalty(true_high) * 10.0
            total_loss = d_loss + r1
            total_loss.backward()
            optimizer_dis.step()

        return fake_high

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:  # Initialize accumulators with first batch
            self.val_features_real = []
            self.val_features_fake = []
            
            # Log real and fake images as before
            real = batch
            resize_transform = transforms.Resize((299, 299), antialias=True)
            real_resized = resize_transform(real)
            
            if self.current_epoch == 0:
                self.logger.experiment.log({
                    "Real": [wandb.Image(
                        real_resized[0].permute(1,2,0).detach().float().cpu().numpy(),
                        caption=" "
                    )]
                })
                self.noise_fixed = torch.randn((10000, self.hparams.latent_dim), device=self.device)

        # Collect features for FID in batches
        if batch_idx * batch.shape[0] < 10000:
            real = batch
            resize_transform = transforms.Resize((299, 299), antialias=True)
            real_resized = resize_transform(real)
            
            # Generate fakes for current batch
            current_noise = self.noise_fixed[batch_idx * batch.shape[0]:(batch_idx + 1) * batch.shape[0]]
            if self.hparams.use_multiscale:
                fake_list = self.generator(current_noise)
                fake = fake_list[-1]  # take the last level
            else:
                fake = self.generator(current_noise)
            fake_resized = resize_transform(fake)

            # Get features through inception
            with torch.no_grad():
                real_features = inception_model(real_resized)
                fake_features = inception_model(fake_resized)
                
                self.val_features_real.append(real_features.cpu())
                self.val_features_fake.append(fake_features.cpu())

        # Calculate FID in the last batch
        if (batch_idx + 1) * batch.shape[0] >= 10000 and self.val_features_real != None:
            # Collect all features
            all_real_features = torch.cat(self.val_features_real, dim=0)[:10000]
            all_fake_features = torch.cat(self.val_features_fake, dim=0)[:10000]
            
            # Calculate statistics
            mu_real = all_real_features.mean(0)
            mu_fake = all_fake_features.mean(0)
            
            cov_real = torch.cov(all_real_features.permute(1,0))
            cov_fake = torch.cov(all_fake_features.permute(1,0))
            
            # Add eps for numerical stability
            eps = 1e-8
            cov_real += eps * torch.eye(cov_real.size(0)).to(cov_real.device)
            cov_fake += eps * torch.eye(cov_fake.size(0)).to(cov_fake.device)
            
            # Calculate FID
            fid_value = calculate_fid(mu_real, cov_real, mu_fake, cov_fake)
            self.log('FID/Validation', fid_value, prog_bar=True, on_epoch=True)
            
            # Clear memory
            del self.val_features_real
            del self.val_features_fake
            torch.cuda.empty_cache()
            self.val_features_real = None
            self.val_features_fake = None
            
            # Log grid as before
            images = []
            for i in range(10):
                for _ in range(5):
                    noise = torch.randn(1, self.hparams.latent_dim, device=self.device)
                    if self.hparams.use_multiscale:
                        generated_list = self.generator(noise)
                        image = generated_list[-1]
                    else:
                        image = self.generator(noise)
                    images.append(image)

            images_resized = [F.interpolate(img, size=(64, 64))[0] for img in images]
            images_grid = torchvision.utils.make_grid(images_resized, nrow=5)
            self.logger.experiment.log({
                "Validation_panel": [wandb.Image(
                    images_grid.permute(1,2,0).detach().float().cpu().numpy(),
                    caption="All"
                )]
            })

    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        optimizer_gen = torch.optim.AdamW(
            self.generator.parameters(),
            lr=lr
        )
        optimizer_disc = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=lr * 0.5 if self.hparams.use_minibatch and self.hparams.type_model != "R1-GAN" and self.hparams.type_model != "LS-GAN" else lr
        )
        scheduler_gen = torch.optim.lr_scheduler.LinearLR(
            optimizer_gen,
            start_factor=1.0,
            end_factor=0.01,
            total_iters=300000
        )
        scheduler_disc = torch.optim.lr_scheduler.LinearLR(
            optimizer_disc,
            start_factor=1.0,
            end_factor=0.01,
            total_iters=300000
        )
        return [
            {
                "optimizer": optimizer_gen,
                "lr_scheduler": {
                    "scheduler": scheduler_gen,
                    "interval": "step"
                }
            },
            {
                "optimizer": optimizer_disc,
                "lr_scheduler": {
                    "scheduler": scheduler_disc,
                    "interval": "step"
                }
            }
        ] 