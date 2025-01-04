import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import nvflare.client as flare_util
import logging
import os

# Replace LitModel import with your actual model class if needed
from custom.models.model_lit import LitModel

# Import your CXP-only DataModule
from data.datamodules.datamodule import CXPDataModule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize NVFlare (sets site name, among other things)
flare_util.init()
SITE_NAME = flare_util.get_site_name()


def main():
    """
    Main function for federated learning with NVFlare and PyTorch Lightning.
    This version is tailored to the CXP dataset only.
    """
    try:
        logger.info(f"Starting NVFlare client for site: {SITE_NAME}")

        # ---------------------------------------------------------------------
        # 1) Initialize the DataModule for the CXP dataset
        # ---------------------------------------------------------------------
        data_module = CXPDataModule(
            cxp_csv_path="/bigdata/andrei_thesis/preprocessed_site_data/CXP/cxp_data.csv",
            cxp_image_dir="/bigdata/andrei_thesis/CXP_data/images",
            batch_size=328,
            num_workers=16,
            pin_memory=True,
        )
        logger.info("CXP DataModule initialized.")

        # Optional setup calls (Lightning will auto-setup on fit/validate anyway)
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
            #    We only have a single val loader now.
            logger.info("Validating on CXP dataset ...")
            trainer.validate(model, data_module.val_dataloader())

            # c) Train on local data
            logger.info("Training locally on CXP dataset ...")
            trainer.fit(model, train_dataloaders=data_module.train_dataloader())

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
