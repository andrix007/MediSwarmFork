from torch.utils.data import DataLoader
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import nvflare.client as flare_util
import logging
import os

# Replace LitModel import with your actual model class if needed
from custom.models.model_lit import LitModel

# Import your new DataModule
from custom.data.datamodules.datamodule import SeparateDatasetDataModule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize NVFlare (sets site name, among other things)
flare_util.init()
SITE_NAME = flare_util.get_site_name()


def main():
    """
    Main function for federated learning with NVFlare and PyTorch Lightning.
    """
    try:
        logger.info(f"Starting NVFlare client for site: {SITE_NAME}")

        # ---------------------------------------------------------------------
        # 1) Initialize the DataModule for local dataset
        #    In a real multi-site setup, each site would ONLY load its own data.
        #    But here, you have a Single DataModule that references all CSVs.
        # ---------------------------------------------------------------------
        data_module = SeparateDatasetDataModule(
            cxp_csv_path="/bigdata/andrei_thesis/preprocessed_site_data/CXP/cxp_data.csv",
            cxp_image_dir="/bigdata/andrei_thesis/CXP_data/images",
            mimic_csv_path="/bigdata/andrei_thesis/preprocessed_site_data/MIMIC/mimic_data.csv",
            mimic_image_dir="/bigdata/andrei_thesis/MIMIC/mimic-cxr-2.0.0.physionet.org",
            nih_csv_path="/bigdata/andrei_thesis/preprocessed_site_data/NIH/nih_data.csv",
            nih_image_dir="/bigdata/andrei_thesis/NIH_data/images",
            batch_size=32,
            num_workers=1,
            pin_memory=True,
        )
        logger.info("Data module initialized.")

        # Optional explicit setup calls
        data_module.setup("fit")
        data_module.setup("validate")

        # ---------------------------------------------------------------------
        # 2) Initialize your Lightning Model
        # ---------------------------------------------------------------------
        model = LitModel(
            lr=0.0005,
            criterion_name="BCELoss",
            num_labels=8,
            in_ch=3,
            out_ch=8,
            spatial_dims=2,
        )
        logger.info("Model initialized.")

        # ---------------------------------------------------------------------
        # 3) Configure the Trainer (GPU/CPU, precision, logging, etc.)
        # ---------------------------------------------------------------------
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
        logger.info("Trainer initialized.")

        # ---------------------------------------------------------------------
        # 4) Federated training loop WITHOUT flare_util.patch(trainer)
        #
        #    We'll handle receiving the global model & sending updates manually.
        # ---------------------------------------------------------------------
        while flare_util.is_running():
            # a) Receive global model or signals from NVFlare
            input_model = flare_util.receive()
            if not input_model:
                logger.info("No input model received — possibly shutting down.")
                break

            round_num = getattr(input_model, "current_round", None)
            logger.info(f"Current round: {round_num}")

            # If the server sent a 'weights' dict, load it
            if hasattr(input_model, "weights"):
                logger.info("Loading global weights into local model.")
                model.load_state_dict(input_model.weights)

            # b) Validate on local data
            #    This data_module returns a dict of val loaders, so we iterate.
            val_loaders = data_module.val_dataloader()
            for dataset_name, val_loader in val_loaders.items():
                logger.info(f"Validating on dataset: {dataset_name} ...")
                trainer.validate(model, val_loader)

            # c) Train on local data
            train_loaders = data_module.train_dataloader()
            for dataset_name, train_loader in train_loaders.items():
                logger.info(f"Training locally on dataset: {dataset_name} ...")
                trainer.fit(
                    model,
                    train_dataloaders=train_loader,
                    val_dataloaders=None,
                )

            # d) Send updated weights back to the server
            flare_util.send({"weights": model.state_dict()})

        # ---------------------------------------------------------------------
        # 5) Save the best checkpoint
        # ---------------------------------------------------------------------
        model.save_best_checkpoint(
            trainer.logger.log_dir, checkpointing.best_model_path
        )
        logger.info("Federated training completed successfully.")

    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise


if __name__ == "__main__":
    main()
