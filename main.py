"""
KL-GAN with Multiple Adversarial Objectives
--------------------------------------------
This script showcases training KL-GAN on CelebA with options for:
    KL-GAN, LS-GAN, WGAN-GP, Hinge-GAN, R1-GAN.

We run 5 different seeds for each method.
"""
import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBar, RichProgressBarTheme

# Import our modules
from models.trainer import GAN_Training
from data.dataset import DataModule
from configs.config import DEFAULT_CONFIG, MODEL_CONFIGS, SEEDS, WANDB_CONFIG

# Set computation precision
torch.set_float32_matmul_precision('medium')

if __name__ == "__main__":
    wandb.login(key="") #key="..."

    # Create dark theme for progress bar
    CustomProgressBar = RichProgressBar(
        refresh_rate=20,
        theme=RichProgressBarTheme(
            description="green_yellow",
            progress_bar="green1",
            progress_bar_finished="green1",
            progress_bar_pulse="#6206E0",
            batch_progress="green_yellow",
            time="grey82",
            processing_speed="grey82",
            metrics="grey82",
        )
    )

    # Run experiments for different methods and seeds
    for method in MODEL_CONFIGS.keys():
        for seed in SEEDS:
            # Current model parameters
            model_config = MODEL_CONFIGS[method]
            use_minibatch = model_config["use_minibatch"]
            use_multiscale = model_config["use_multiscale"]

            # Create run name
            minibatch_suffix = "with_minibatch" if use_minibatch else "no_minibatch"
            scale_suffix = "multiscale" if use_multiscale else "single_scale"
            run_name = f"{method}_{minibatch_suffix}_{scale_suffix}"
            
            # Create W&B logger
            wandb_logger = WandbLogger(
                name=run_name,
                project=WANDB_CONFIG["project"],
                save_dir=WANDB_CONFIG["save_dir"],
                version=None,
                reinit=True
            )

            # Create PyTorch Lightning trainer
            trainer = pl.Trainer(
                accelerator="gpu",
                devices="auto",
                precision="bf16-mixed",
                log_every_n_steps=DEFAULT_CONFIG["log_every_n_steps"],
                callbacks=[CustomProgressBar],
                logger=[wandb_logger],
                max_epochs=DEFAULT_CONFIG["max_epochs"],
                limit_train_batches=1.0,
                check_val_every_n_epoch=DEFAULT_CONFIG["check_val_every_n_epoch"]
            )

            # Create model
            model = GAN_Training(
                learning_rate=DEFAULT_CONFIG["learning_rate"],
                batch_size=DEFAULT_CONFIG["batch_size"],
                seed_value=seed,
                type_model=method,
                latent_dim=DEFAULT_CONFIG["latent_dim"],
                dim=DEFAULT_CONFIG["dim"],
                use_minibatch=use_minibatch,
                use_multiscale=use_multiscale,
            )

            # Create data module
            datamodule = DataModule(
                batch_size=model.hparams.batch_size,
                val_batch_size=1024,
                data_dir="./img_align_celeba/img_align_celeba"
            )

            # Start training
            trainer.fit(model, datamodule)
            
            # Close current experiment
            wandb_logger.experiment.finish() 