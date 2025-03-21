"""
Configuration file for KL-GAN experiments
"""

# Common parameters for all experiments
DEFAULT_CONFIG = {
    "learning_rate": 0.00008,
    "batch_size": 1024,
    "latent_dim": 128,
    "dim": 128,
    "log_every_n_steps": 40,
    "max_epochs": 300,
    "check_val_every_n_epoch": 50
}

# Parameters for different model types
MODEL_CONFIGS = {
    "KL-GAN": {
        "use_minibatch": True,
        "use_multiscale": False,
    },
    "R1-GAN": {
        "use_minibatch": True,
        "use_multiscale": False,
    },
    "LS-GAN": {
        "use_minibatch": True,
        "use_multiscale": False,
    },
    "WGAN-GP": {
        "use_minibatch": True,
        "use_multiscale": False,
    },
    "Hinge-GAN": {
        "use_minibatch": True,
        "use_multiscale": False,
    }
}

# Parameters for different seeds
SEEDS = [1, 2, 3, 4, 5]

# Weights & Biases configuration
WANDB_CONFIG = {
    "project": "KL-GAN CelebA Experiment",
    "save_dir": "./wandb_logs"
} 