import os
import cv2
import torch
import albumentations as A
import albumentations.pytorch as AP
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader


class CelebADataset(Dataset):
    """
    PyTorch dataset wrapper for preloaded GPU tensors of images.
    """
    def __init__(self, data_tensor):
        self.data = data_tensor

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx]


class DataModule(pl.LightningDataModule):
    """
    DataModule for CelebA dataset. Adjusts transforms and splits into train/val.
    """
    def __init__(self, batch_size, val_batch_size, data_dir="./img_align_celeba/img_align_celeba"):
        super().__init__()
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.data_dir = data_dir

        self.train_transform = A.Compose([
            A.Resize(32, 32),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            AP.ToTensorV2(),
        ])

        self.val_transform = A.Compose([
            A.Resize(32, 32),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            AP.ToTensorV2(),
        ])

    def setup(self, stage=None):
        image_files = os.listdir(self.data_dir)
        images = []

        for img_file in image_files:
            image_path = os.path.join(self.data_dir, img_file)
            image = cv2.imread(image_path)
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.train_transform(image=image)["image"]
            images.append(image)

        data_tensor = torch.stack(images)
        data_tensor = data_tensor.to(device=torch.device('cuda'), dtype=torch.bfloat16)

        self.dataset = CelebADataset(data_tensor)
        split = int(0.9 * len(self.dataset))
        self.dataset_train, self.dataset_val = torch.utils.data.random_split(
            self.dataset, [split, len(self.dataset) - split]
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=0,
            pin_memory=False
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val, 
            batch_size=self.val_batch_size, 
            shuffle=False, 
            num_workers=0,
            pin_memory=False
        )


class LatentDataset(Dataset):
    """
    Stores latent codes from *.pt files in a specified directory.
    """
    def __init__(self, latent_dir):
        self.latent_dir = latent_dir
        self.latent_files = [f for f in os.listdir(latent_dir) if f.endswith('.pt')]

    def __len__(self):
        return len(self.latent_files)

    def __getitem__(self, idx):
        latent_path = os.path.join(self.latent_dir, self.latent_files[idx])
        latent = torch.load(latent_path)
        return latent


class LatentDataModule(pl.LightningDataModule):
    """
    DataModule for loading latent *.pt files.
    """
    def __init__(self, batch_size, val_batch_size, latent_dir):
        super().__init__()
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.latent_dir = latent_dir

    def setup(self, stage=None):
        self.dataset_train = LatentDataset(latent_dir=self.latent_dir)
        self.dataset_val = LatentDataset(latent_dir=self.latent_dir)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.val_batch_size, shuffle=False, num_workers=8) 