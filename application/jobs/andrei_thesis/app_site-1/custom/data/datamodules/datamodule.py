import pytorch_lightning as pl
from torch.utils.data import DataLoader
from custom.data.datasets.all_datasets import AllDatasetsShared
from torchvision import transforms
import pandas as pd


class CXPDataModule(pl.LightningDataModule):
    """
    LightningDataModule specifically for CXP dataset.
    """

    def __init__(
        self,
        cxp_csv_path: str,
        cxp_image_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        """
        Initializes the DataModule with dataset parameters.

        Args:
            cxp_csv_path (str): Path to the CSV file for CXP (e.g. "cxp_data.csv").
            cxp_image_dir (str): Path to the image directory for CXP.
            batch_size (int, optional): Batch size. Defaults to 32.
            num_workers (int, optional): Number of workers for data loading. Defaults to 4.
            pin_memory (bool, optional): Pin memory for data transfer to GPU. Defaults to True.
        """
        super().__init__()
        self.cxp_csv_path = cxp_csv_path
        self.cxp_image_dir = cxp_image_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # Transforms
        self.transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Placeholders for subsets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        """
        Sets up the dataset for each stage (fit, validate, test).

        Args:
            stage (str, optional): 'fit', 'validate', or 'test'. Defaults to None.
        """
        if stage in ("fit", None):
            # Load training subset
            df_train = self._load_csv(self.cxp_csv_path, split="train")
            self.train_dataset = AllDatasetsShared(
                dataframe=df_train,
                transform=self.transforms
            )

            # Load validation subset
            df_val = self._load_csv(self.cxp_csv_path, split="val")
            self.val_dataset = AllDatasetsShared(
                dataframe=df_val,
                transform=self.transforms
            )

        if stage in ("test", None):
            # Load test subset
            df_test = self._load_csv(self.cxp_csv_path, split="test")
            self.test_dataset = AllDatasetsShared(
                dataframe=df_test,
                transform=self.transforms
            )

    def train_dataloader(self):
        """
        Returns a single train DataLoader for the CXP dataset.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory
        )

    def val_dataloader(self):
        """
        Returns a single validation DataLoader for the CXP dataset.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory
        )

    def test_dataloader(self):
        """
        Returns a single test DataLoader for the CXP dataset.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory
        )

    def _load_csv(self, csv_path, split: str):
        """
        Loads a CSV for a specific split. Assumes the naming scheme
        uses 'training_data.csv', 'val_data.csv', 'test_data.csv' or similar.

        Args:
            csv_path (str): Path to the main CSV (like 'cxp_data.csv').
            split (str): 'train', 'val', or 'test'.

        Returns:
            pd.DataFrame: DataFrame for the specified split.
        """
        if split == "train":
            return pd.read_csv(csv_path.replace("data.csv", "training_data.csv"))
        elif split == "val":
            return pd.read_csv(csv_path.replace("data.csv", "val_data.csv"))
        elif split == "test":
            return pd.read_csv(csv_path.replace("data.csv", "test_data.csv"))
        else:
            raise ValueError(f"Invalid split: {split}")
