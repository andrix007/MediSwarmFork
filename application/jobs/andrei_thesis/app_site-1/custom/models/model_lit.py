import pandas as pd
import torch
from torch import nn
import pytorch_lightning as pl
from torchvision.models import densenet121, DenseNet121_Weights
from .base_model import BasicClassifier
import logging

logger = logging.getLogger(__name__)


class LitModel(BasicClassifier):
    """
    A PyTorch Lightning model based on DenseNet121 for classification tasks.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        spatial_dims: int,
        model_type="densenet",
        lr=0.0005,
        criterion_name="BCELoss",
        num_labels=8,
        seed=-1,
    ):
        """
        Initializes the LitModel.

        Args:
            in_ch (int): Number of input channels.
            out_ch (int): Number of output classes.
            spatial_dims (int): Spatial dimensions for the input (e.g., 2D or 3D).
            model_type (str): Type of the model ('densenet' or 'resume').
            lr (float): Learning rate for the optimizer.
            criterion_name (str): Name of the loss function.
            num_labels (int): Number of labels for classification.
            seed (int): Seed for reproducibility.
        """
        super().__init__(in_ch, out_ch, spatial_dims, optimizer=torch.optim.Adam, optimizer_kwargs={"lr": lr})
        self.lr = lr
        self.num_labels = num_labels
        self.seed = seed
        self.epoch_logs = []  # To track epoch logs
        self.current_train_loss = None
        self.current_lr = lr

        # Initialize the model
        if model_type == "densenet":
            self.model = densenet121(weights=DenseNet121_Weights.DEFAULT)
            num_ftrs = self.model.classifier.in_features
            # self.model.classifier = nn.Sequential(
            #     nn.Linear(num_ftrs, out_ch),  # Use `out_ch` for the output size
            #     nn.Sigmoid() if criterion_name == "BCELoss" else nn.Identity()
            # )
            self.model.classifier = nn.Linear(num_ftrs, out_ch)
            self.criterion = nn.BCEWithLogitsLoss()

        elif model_type == "resume":
            checkpoint = torch.load("results/checkpoint")
            self.model = checkpoint["model"]
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        # Define the loss function
        self.criterion = nn.BCEWithLogitsLoss() if criterion_name == "BCELoss" else nn.CrossEntropyLoss()

        logger.info(f"Total Trainable Parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")

    def forward(self, x):
        """
        Forward pass for the model.

        Args:
            x (torch.Tensor): Input tensor.
        """
        return self.model(x)

    def _shared_step(self, batch, batch_idx, state):
        """
        Shared logic for train, validation, and test steps.

        Args:
            batch: A batch of data.
            batch_idx (int): Index of the batch.
            state (str): State of the model ('train', 'val', or 'test').
        """
        imgs, labels = batch[:2]
        preds = self(imgs)
        loss = self.criterion(preds, labels)

        # Log metrics and loss
        if state == "train":
            accuracy = self.acc["train_"].update(preds, labels)

            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            #self.log("train_auc", auc_roc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        elif state == "val":

            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            logger.info(f"[DEBUG] Validation Loss: {loss.item()}")

            #self.log("val_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        imgs, labels, paths = batch
        preds = self(imgs)
        loss = self.criterion(preds, labels)

        accuracy = self.acc["test_"].update(preds, labels)
        auc_roc = self.auc_roc["test_"].update(preds, labels)

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self.log("test_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self.log("test_auc", auc_roc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {"preds": preds, "labels": labels, "paths": paths}

    def on_train_epoch_end(self):
        optimizer = self.trainer.optimizers[0]
        self.current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"Epoch {self.current_epoch} completed. Current LR: {self.current_lr}")

    def on_validation_epoch_end(self):
        metrics = self.trainer.callback_metrics

        logger.info(f"[DEBUG] Callback Metrics at Validation Epoch End: {metrics}")

        val_loss = metrics.get("val_loss", None)
        val_accuracy = metrics.get("val_accuracy", None)

        logger.info(f"Validation metrics: Loss={val_loss}, Accuracy={val_accuracy}")

        epoch_log = {
            "epoch": self.current_epoch,
            "train_loss": self.current_train_loss.item() if self.current_train_loss is not None else "N/A",
            "val_loss": val_loss.item() if val_loss is not None else "N/A",
            "val_accuracy": val_accuracy.item() if val_accuracy is not None else "N/A",
            "seed": self.seed,
            "lr": self.current_lr,
        }
        self.epoch_logs.append(epoch_log)
        pd.DataFrame(self.epoch_logs).to_csv("epoch_logs.csv", index=False)

    def on_test_epoch_start(self):
        self.test_outputs = []

    def on_test_epoch_end(self):
        all_preds = torch.cat([x["preds"] for x in self.test_outputs], dim=0)
        all_labels = torch.cat([x["labels"] for x in self.test_outputs], dim=0)
        all_paths = sum([x["paths"] for x in self.test_outputs], [])  # Concatenate paths

        torch.save({"preds": all_preds, "labels": all_labels, "paths": all_paths}, "test_predictions.pt")
        self.test_outputs = []  # Free memory
