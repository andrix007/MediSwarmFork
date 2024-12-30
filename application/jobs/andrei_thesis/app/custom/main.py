from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from collections import Counter
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import nvflare.client as flare_util
from custom.models.model_lit import LitModel
from custom.data.datamodules.datamodule import LitDataModule
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize NVFlare
flare_util.init()

SITE_NAME = flare_util.get_site_name()

def main():
    """
    Main function for federated learning with NVFlare and PyTorch Lightning.
    """
    try:
        logger.info(f"Starting NVFlare client for site: {SITE_NAME}")

        # Initialize data module
        data_module = LitDataModule(
            train_csv="/path/to/train.csv",
            val_csv="/path/to/val.csv",
            test_csv="/path/to/test.csv",
            image_dir="/path/to/images",
            batch_size=32,
        )
        logger.info("Data module initialized")

        # Prepare dataset and split into train/validation
        dataset = data_module.prepare_dataset()
        labels = dataset.get_labels()

        indices = list(range(len(dataset)))
        train_indices, val_indices = train_test_split(
            indices, test_size=0.2, stratify=labels, random_state=42
        )
        ds_train = Subset(dataset, train_indices)
        ds_val = Subset(dataset, val_indices)

        logger.info(f"Training size: {len(ds_train)}, Validation size: {len(ds_val)}")

        # Initialize the model
        model = LitModel(
            lr=0.0005,
            criterion_name="BCELoss",
            num_labels=8,
            in_ch=3,
            out_ch=8,
            spatial_dims=2,
        )
        logger.info("Model initialized")

        # Configure trainer
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using {accelerator} for training")

        path_run_dir = f"./runs/{SITE_NAME}"
        os.makedirs(path_run_dir, exist_ok=True)

        checkpointing = ModelCheckpoint(
            dirpath=path_run_dir,
            monitor="val_loss",
            save_last=True,
            save_top_k=1,
            mode="min",
        )

        trainer = Trainer(
            accelerator=accelerator,
            precision=16 if torch.cuda.is_available() else 32,
            default_root_dir=path_run_dir,
            callbacks=[checkpointing],
            enable_checkpointing=True,
            check_val_every_n_epoch=1,
            log_every_n_steps=10,
            max_epochs=10,
            logger=TensorBoardLogger(save_dir=path_run_dir),
        )
        logger.info("Trainer initialized")

        # Patch trainer for federated learning
        flare_util.patch(trainer)

        # Federated training loop
        while flare_util.is_running():
            input_model = flare_util.receive()
            logger.info(f"Current round: {input_model.current_round}")

            # Validate global model
            trainer.validate(model, dataloaders=DataLoader(ds_val, batch_size=32))

            # Train local model
            trainer.fit(model, dataloaders=DataLoader(ds_train, batch_size=32))

        # Save the best checkpoint
        model.save_best_checkpoint(
            trainer.logger.log_dir, checkpointing.best_model_path
        )
        logger.info("Training completed successfully")

    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise


if __name__ == "__main__":
    main()
