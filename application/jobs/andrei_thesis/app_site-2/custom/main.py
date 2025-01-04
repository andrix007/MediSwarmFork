import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import nvflare.client as flare_util
import logging
import os

# If needed, replace with your model import path
from custom.models.model_lit import LitModel

# Import your MIMIC-only DataModule
from data.datamodules.datamodule import MIMICDataModule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize NVFlare (sets site name, etc.)
flare_util.init()
SITE_NAME = flare_util.get_site_name()


def main():
    """
    Main function for federated learning with NVFlare and PyTorch Lightning (MIMIC dataset).
    """
    try:
        logger.info(f"Starting NVFlare client for site: {SITE_NAME}")

        # 1) Initialize the MIMIC DataModule
        data_module = MIMICDataModule(
            mimic_csv_path="/bigdata/andrei_thesis/preprocessed_site_data/MIMIC/mimic_data.csv",
            mimic_image_dir="/bigdata/andrei_thesis/MIMIC/mimic-cxr-2.0.0.physionet.org",
            batch_size=328,
            num_workers=16,
            pin_memory=True
        )
        logger.info("MIMIC DataModule initialized.")

        data_module.setup("fit")
        data_module.setup("validate")

        # 2) Initialize the Lightning Model
        model = LitModel(
            lr=0.0005,
            criterion_name="BCELoss",
            num_labels=8,
            in_ch=3,
            out_ch=8,
            spatial_dims=2,
        )
        logger.info("Model initialized.")

        # 3) Configure the Trainer
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

        # 4) Federated training loop
        while flare_util.is_running():
            input_model = flare_util.receive()
            if not input_model:
                logger.info("No input model received — possibly shutting down.")
                break

            round_num = getattr(input_model, "current_round", None)
            logger.info(f"Current round: {round_num}")

            # Load global weights if provided
            if hasattr(input_model, "weights"):
                logger.info("Loading global weights into local model.")
                model.load_state_dict(input_model.weights)

            # Validate locally
            logger.info("Validating on MIMIC dataset ...")
            trainer.validate(model, data_module.val_dataloader())

            # Train locally
            logger.info("Training locally on MIMIC dataset ...")
            trainer.fit(model, train_dataloaders=data_module.train_dataloader())

            # Send updates back to server
            flare_util.send({"weights": model.state_dict()})

        # 5) Save best checkpoint
        model.save_best_checkpoint(
            trainer.logger.log_dir, checkpointing.best_model_path
        )
        logger.info("Federated training completed successfully (MIMIC).")

    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise


if __name__ == "__main__":
    main()
