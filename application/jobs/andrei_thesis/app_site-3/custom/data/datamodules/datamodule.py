import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from custom.data.datasets.all_datasets import AllDatasetsShared
from torchvision import transforms
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class NIHDataModule(pl.LightningDataModule):
    """
    LightningDataModule for the NIH dataset only.
    """

    def __init__(
        self,
        nih_csv_path: str,
        nih_image_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        """
        Args:
            nih_csv_path (str): Path to the CSV file for NIH (e.g. "nih_data.csv").
            nih_image_dir (str): Path to the image directory for NIH.
            batch_size (int, optional): Batch size. Defaults to 32.
            num_workers (int, optional): Number of workers for data loading. Defaults to 4.
            pin_memory (bool, optional): Pin memory for data transfer to GPU. Defaults to True.
        """
        super().__init__()
        self.nih_csv_path = nih_csv_path
        self.nih_image_dir = nih_image_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Placeholders
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        """
        Sets up the dataset for each stage (fit, validate, test).
        """

        if stage in ("fit", None):
            df_train = self._load_csv(self.nih_csv_path, split="train")
            self.train_dataset = AllDatasetsShared(
                dataframe=df_train,
                transform=self.transforms
            )
            logger.info(f"Train Dataset: {len(self.train_dataset)} samples")

            df_val = self._load_csv(self.nih_csv_path, split="val")
            self.val_dataset = AllDatasetsShared(
                dataframe=df_val,
                transform=self.transforms
            )
            logger.info(f"Validation Dataset: {len(self.val_dataset)} samples")

        if stage in ("test", None):
            df_test = self._load_csv(self.nih_csv_path, split="test")
            self.test_dataset = AllDatasetsShared(
                dataframe=df_test,
                transform=self.transforms
            )
            logger.info(f"Test Dataset: {len(self.test_dataset)} samples")
        


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory
        )

    def _load_csv(self, csv_path, split: str):
        """
        Loads a CSV for a specific split, replacing 'data.csv' with
        'training_data.csv', 'val_data.csv', 'test_data.csv'.

        Args:
            csv_path (str): e.g. 'nih_data.csv'
            split (str): 'train', 'val', or 'test'

        Returns:
            pd.DataFrame
        """
        if split == "train":
            return pd.read_csv(csv_path.replace("data.csv", "training_data.csv"))
        elif split == "val":
            return pd.read_csv(csv_path.replace("data.csv", "val_data.csv"))
        elif split == "test":
            return pd.read_csv(csv_path.replace("data.csv", "test_data.csv"))
        else:
            raise ValueError(f"Invalid split: {split}")
