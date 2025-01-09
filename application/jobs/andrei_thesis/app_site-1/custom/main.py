import time
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import nvflare.client as flare_util
import logging
import os

# Replace LitModel import with your actual model class if needed
from custom.models.model_lit import LitModel

from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.apis.dxo import DXO, MetaKey, DataKind

# Import your CXP-only DataModule
from data.datamodules.datamodule import CXPDataModule

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
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
            batch_size=128,
            num_workers=8,
            pin_memory=True,
        )
        logger.info(f"CXP CSV Path: {data_module.cxp_csv_path}")
        logger.info(f"CXP Image Directory: {data_module.cxp_image_dir}")
        logger.info(f"Batch size: {data_module.batch_size}, Num workers: {data_module.num_workers}")
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
        if torch.cuda.is_available():
            logger.info(f"GPU Utilization: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")

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
            max_epochs=1,
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
            
            logger.info("Received input model with current round: %s", round_num)
            logger.info("Training batch size: %s", len(data_module.train_dataloader()))

            # Load global weights if provided or initialize weights
            if round_num == 0:
                if hasattr(input_model, "weights"):
                    logger.info("Initializing model with global weights for round 0.")
                    model.load_state_dict(input_model.weights)
                else:
                    logger.warning("No global weights received for round 0. Using random initialization.")
            elif hasattr(input_model, "weights"):
                logger.info(f"Updating local model with received global weights for round {round_num} and NIH dataset.")
                model.load_state_dict(input_model.weights)

            # Validate locally
            start_time = time.time()
            try:
                logger.info("Validating on CXP dataset ...")
                trainer.validate(model, data_module.val_dataloader())
                logger.info("CXP Validation completed successfully.")
                logger.info(f"CXP Validation completed in {time.time() - start_time:.2f} seconds.")
                logger.info(f"Validation Metrics: {trainer.callback_metrics}")
                validation_metrics = {
                    "val_loss": trainer.callback_metrics.get("val_loss", torch.tensor(float('inf'))).item(),
                }
            except Exception as e:
                logger.error(f"CXP Validation failed: {e}")
                logger.error(f"CXP Validation failed in {time.time() - start_time:.2f} seconds.")
                raise

            # Train locally
            start_time = time.time()
            try:
                logger.info("Training locally on CXP dataset ...")
                trainer.fit(model, train_dataloaders=data_module.train_dataloader())
                logger.info("CXP Training completed successfully.")
                logger.info(f"CXP Training completed in {time.time() - start_time:.2f} seconds.")
                logger.info(f"Training Metrics: {trainer.callback_metrics}")
                training_metrics = {
                    "train_loss": trainer.callback_metrics.get("train_loss", torch.tensor(float('inf'))).item(),
                    "train_loss_epoch": trainer.callback_metrics.get("train_loss_epoch", torch.tensor(float('inf'))).item(),
                }
            except Exception as e:
                logger.error(f"CXP Training failed: {e}")
                logger.error(f"CXP Training failed in {time.time() - start_time:.2f} seconds.")
                raise

            # d) Send updated weights back to the server
            state_dict = model.state_dict()

            num_steps = len(data_module.train_dataloader())/data_module.batch_size  # Or however you compute steps

            fl_model = FLModel(
                params=state_dict,  # Model weights
                metrics={**validation_metrics, **training_metrics},  # Combine validation and training metrics
                meta={
                    MetaKey.NUM_STEPS_CURRENT_ROUND: num_steps,
                    MetaKey.CURRENT_ROUND: round_num,
                },
            )

            try:
                logger.info("Sending model weights and metrics to the server ...")
                flare_util.send(fl_model)
                logger.info(f"Metrics sent to server: {fl_model.metrics}")
            except Exception as e:
                logger.error(f"Failed to send model weights/metrics: {e}")
                raise

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
