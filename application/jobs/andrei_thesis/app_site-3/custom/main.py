import time
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import nvflare.client as flare_util
import logging
import os

from custom.models.model_lit import LitModel
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.apis.dxo import DXO, MetaKey, DataKind
from nvflare.client.utils import numerical_params_diff

# NIH-only DataModule
from data.datamodules.datamodule import NIHDataModule

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

flare_util.init()
SITE_NAME = flare_util.get_site_name()

def main():
    """
    Main function for federated learning with NVFlare and PyTorch Lightning (NIH dataset).
    Shows how to force a fresh train each round.
    """
    try:
        logger.info(f"Starting NVFlare client for site: {SITE_NAME}")

        # 1) Initialize your DataModule
        data_module = NIHDataModule(
            nih_csv_path="/bigdata/andrei_thesis/preprocessed_site_data/NIH/nih_data.csv",
            nih_image_dir="/bigdata/andrei_thesis/NIH_data/images",
            batch_size=172,
            num_workers=8,
            pin_memory=True
        )
        logger.info(f"NIH CSV Path: {data_module.nih_csv_path}")
        logger.info(f"NIH Image Directory: {data_module.nih_image_dir}")
        logger.info(f"Batch size: {data_module.batch_size}, Num workers: {data_module.num_workers}")
        logger.info("NIH DataModule initialized.")

        # 2) Create a function to build a new Trainer each round
        def make_trainer(run_dir):
            checkpointing = ModelCheckpoint(
                dirpath=run_dir,
                monitor="val_loss",
                save_last=True,
                save_top_k=1,
                mode="min",
            )

            accelerator = "gpu" if torch.cuda.is_available() else "cpu"
            trainer = Trainer(
                accelerator=accelerator,
                precision=16 if torch.cuda.is_available() else 32,
                default_root_dir=run_dir,
                callbacks=[checkpointing],
                enable_checkpointing=True,
                check_val_every_n_epoch=1,
                log_every_n_steps=10,
                # IMPORTANT: only 1 epoch per round:
                max_epochs=1,
                logger=TensorBoardLogger(save_dir=run_dir),
            )
            return trainer, checkpointing

        # 3) Initialize the model once
        model = LitModel(
            lr=0.0005,
            criterion_name="BCELoss",
            num_labels=8,
            in_ch=3,
            out_ch=8,
            spatial_dims=2,
        )
        logger.info(f"Model initialized with learning rate: {model.lr}, criterion: {model.criterion}")

        # 4) Federated training loop
        path_run_dir = f"./runs/{SITE_NAME}"
        os.makedirs(path_run_dir, exist_ok=True)

        current_round = 0
        while flare_util.is_running():
            # ------------------------------------------------------------------
            # 4a) Get aggregator's updated model (or None if no more rounds).
            input_model = flare_util.receive()
            if not input_model:
                logger.info("No input model received — possibly shutting down.")
                break

            original_params = input_model.params
            round_num = getattr(input_model, "current_round", None)
            if round_num is None:
                # Fallback if aggregator didn't set 'current_round'
                round_num = current_round
            logger.info(f"Received input model with current round: {round_num}")

            # ------------------------------------------------------------------
            # 4b) Load aggregator's global weights
            if round_num == 0:
                if hasattr(input_model, "weights"):
                    logger.info("Round 0: Initializing model with global weights.")
                    model.load_state_dict(input_model.weights)
                else:
                    logger.warning("Round 0: No global weights received.")
            else:
                if hasattr(input_model, "weights"):
                    logger.info(f"Round {round_num}: Updating local model with aggregator's weights.")
                    model.load_state_dict(input_model.weights)

            # ------------------------------------------------------------------
            # 4c) Create a fresh Trainer and run validation/training
            #     so that each round does a real epoch
            round_run_dir = os.path.join(path_run_dir, f"round_{round_num}")
            os.makedirs(round_run_dir, exist_ok=True)

            trainer, checkpointing = make_trainer(round_run_dir)

            # 4d) Re-run setup("fit") to ensure DataModule is re-initialized for a new round
            data_module.setup("fit")
            data_module.setup("validate")

            # 4e) Validate locally (optional but recommended)
            logger.info(f"Validating locally for round {round_num} ...")
            val_start = time.time()
            results = trainer.validate(model, data_module.val_dataloader())
            logger.info(f"[DEBUG] Validation Results for round {round_num}: {results}")
            logger.info(f"Validation took {time.time() - val_start:.2f} seconds.")
            logger.info(f"Validation callback metrics: {trainer.callback_metrics}")
            validation_metrics = {
                "val_loss": trainer.callback_metrics.get("val_loss", torch.tensor(float('inf'))).item(),
                "val_accuracy": trainer.callback_metrics.get("val_accuracy", torch.tensor(0.0)).item(),
            }

            # ------------------------------------------------------------------
            # 4f) Train locally for this round
            logger.info(f"Training locally on round {round_num} ...")
            train_start = time.time()

            # Re-run setup("fit") to ensure fresh shuffle, etc.
            data_module.setup("fit")
            trainer.fit(model, train_dataloaders=data_module.train_dataloader())

            logger.info(f"Training completed in {time.time() - train_start:.2f} seconds.")
            logger.info(f"Training callback metrics: {trainer.callback_metrics}")
            training_metrics = {
                "train_loss": trainer.callback_metrics.get("train_loss", torch.tensor(float('inf'))).item(),
                "train_accuracy": trainer.callback_metrics.get("train_accuracy", torch.tensor(0.0)).item(),
            }

            # ------------------------------------------------------------------
            # 4g) Prepare the weight diff or full weights to send to the aggregator
            new_params = model.state_dict()
            num_steps = len(data_module.train_dataloader()) / data_module.batch_size

            # Make a DXO with WEIGHT_DIFF or WEIGHTS
            model_dxo = DXO(data_kind=DataKind.WEIGHT_DIFF, data=new_params)
            # Put metrics in meta so aggregator can do "best model" selection
            model_dxo.set_meta_prop(
                MetaKey.INITIAL_METRICS,
                {
                    "val_loss": validation_metrics["val_loss"],
                    "val_accuracy": validation_metrics["val_accuracy"],
                    "train_loss": training_metrics["train_loss"],
                    "train_accuracy": training_metrics["train_accuracy"],
                }
            )
            logger.info(f"[DEBUG] DXO meta props: {model_dxo.get_meta_props()}")
            #logger.info(f"[DEBUG] DXO shareable content: {model_dxo.to_shareable()}")
            fl_model = FLModel(
                params=model_dxo.to_shareable(),
                metrics={**validation_metrics, **training_metrics},
                meta={
                    MetaKey.NUM_STEPS_CURRENT_ROUND: num_steps,
                    MetaKey.CURRENT_ROUND: round_num,
                },
            )

            # If round 0, send full weights (or you can still do diffs, but typically we start with full).
            if round_num == 0:
                fl_model.params = new_params
            else:
                fl_model.params = numerical_params_diff(original_params, new_params)

            # ------------------------------------------------------------------
            # 4h) Send results back to aggregator
            try:
                logger.info(f"Sending updated weights and metrics to aggregator for round {round_num} ...")
                flare_util.send(fl_model)
                logger.info(f"Metrics sent: {fl_model.metrics}")
            except Exception as e:
                logger.error(f"Failed to send model/metrics: {e}")
                raise

            # # If you want to do a "best checkpoint" each round, do it here
            # model.save_best_checkpoint(
            #     trainer.logger.log_dir, checkpointing.best_model_path
            # )

            # Bump round number (if aggregator doesn't do it)
            current_round += 1

        logger.info("Federated training completed successfully (NIH).")

    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise


if __name__ == "__main__":
    main()
