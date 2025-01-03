import pytorch_lightning as pl
from torch.utils.data import DataLoader
from custom.data.datasets.all_datasets import AllDatasetsShared  # Assuming this is your updated dataset class
from torchvision import transforms

class SeparateDatasetDataModule(pl.LightningDataModule):
    """
    LightningDataModule for handling datasets (CXP, MIMIC, NIH) separately.
    """

    def __init__(
        self,
        cxp_csv_path: str,
        cxp_image_dir: str,
        mimic_csv_path: str,
        mimic_image_dir: str,
        nih_csv_path: str,
        nih_image_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        """
        Initializes the DataModule with datasets and parameters.

        Args:
            cxp_csv_path (str): Path to the CSV file for CXP.
            cxp_image_dir (str): Path to the image directory for CXP.
            mimic_csv_path (str): Path to the CSV file for MIMIC.
            mimic_image_dir (str): Path to the image directory for MIMIC.
            nih_csv_path (str): Path to the CSV file for NIH.
            nih_image_dir (str): Path to the image directory for NIH.
            batch_size (int, optional): Batch size. Defaults to 32.
            num_workers (int, optional): Number of workers for data loading. Defaults to 4.
            pin_memory (bool, optional): Pin memory for data transfer to GPU. Defaults to True.
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.cxp_csv_path = cxp_csv_path
        self.cxp_image_dir = cxp_image_dir
        self.mimic_csv_path = mimic_csv_path
        self.mimic_image_dir = mimic_image_dir
        self.nih_csv_path = nih_csv_path
        self.nih_image_dir = nih_image_dir

        # Transforms
        self.transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Placeholders for datasets
        self.cxp_train, self.cxp_val, self.cxp_test = None, None, None
        self.mimic_train, self.mimic_val, self.mimic_test = None, None, None
        self.nih_train, self.nih_val, self.nih_test = None, None, None

    def setup(self, stage=None):
        """
        Sets up datasets for each stage (fit, validate, test).

        Args:
            stage (str, optional): Current stage (fit, validate, test). Defaults to None.
        """
        if stage == "fit" or stage is None:
            self.cxp_train = AllDatasetsShared(
                dataframe=self._load_csv(self.cxp_csv_path, "train"),
                transform=self.transforms
            )
            self.mimic_train = AllDatasetsShared(
                dataframe=self._load_csv(self.mimic_csv_path, "train"),
                transform=self.transforms
            )
            self.nih_train = AllDatasetsShared(
                dataframe=self._load_csv(self.nih_csv_path, "train"),
                transform=self.transforms
            )

        if stage == "validate" or stage is None:
            self.cxp_val = AllDatasetsShared(
                dataframe=self._load_csv(self.cxp_csv_path, "val"),
                transform=self.transforms
            )
            self.mimic_val = AllDatasetsShared(
                dataframe=self._load_csv(self.mimic_csv_path, "val"),
                transform=self.transforms
            )
            self.nih_val = AllDatasetsShared(
                dataframe=self._load_csv(self.nih_csv_path, "val"),
                transform=self.transforms
            )

        if stage == "test" or stage is None:
            self.cxp_test = AllDatasetsShared(
                dataframe=self._load_csv(self.cxp_csv_path, "test"),
                transform=self.transforms
            )
            self.mimic_test = AllDatasetsShared(
                dataframe=self._load_csv(self.mimic_csv_path, "test"),
                transform=self.transforms
            )
            self.nih_test = AllDatasetsShared(
                dataframe=self._load_csv(self.nih_csv_path, "test"),
                transform=self.transforms
            )

    def train_dataloader(self):
        """
        Returns separate training dataloaders for each dataset.
        """
        return {
            "cxp": DataLoader(
                self.cxp_train,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=True,
                pin_memory=self.pin_memory
            ),
            "mimic": DataLoader(
                self.mimic_train,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=True,
                pin_memory=self.pin_memory
            ),
            "nih": DataLoader(
                self.nih_train,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=True,
                pin_memory=self.pin_memory
            )
        }

    def val_dataloader(self):
        """
        Returns separate validation dataloaders for each dataset.
        """
        return {
            "cxp": DataLoader(
                self.cxp_val,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                pin_memory=self.pin_memory
            ),
            "mimic": DataLoader(
                self.mimic_val,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                pin_memory=self.pin_memory
            ),
            "nih": DataLoader(
                self.nih_val,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                pin_memory=self.pin_memory
            )
        }

    def test_dataloader(self):
        """
        Returns separate test dataloaders for each dataset.
        """
        return {
            "cxp": DataLoader(
                self.cxp_test,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                pin_memory=self.pin_memory
            ),
            "mimic": DataLoader(
                self.mimic_test,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                pin_memory=self.pin_memory
            ),
            "nih": DataLoader(
                self.nih_test,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                pin_memory=self.pin_memory
            )
        }

    def _load_csv(self, csv_path, split):
        """
        Utility function to load a CSV for a specific split.

        Args:
            csv_path (str): Path to the CSV file.
            split (str): Split type (train, val, test).

        Returns:
            pd.DataFrame: DataFrame for the split.
        """
        import pandas as pd

        if split == "train":
            return pd.read_csv(csv_path.replace("data.csv", "training_data.csv"))
        elif split == "val":
            return pd.read_csv(csv_path.replace("data.csv", "val_data.csv"))
        elif split == "test":
            return pd.read_csv(csv_path.replace("data.csv", "test_data.csv"))
        else:
            raise ValueError(f"Invalid split: {split}")
