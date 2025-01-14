import pandas as pd
import torch
from torch import nn
import pytorch_lightning as pl
from torchvision.models import densenet121, DenseNet121_Weights
from torchmetrics import Accuracy, AUROC
import logging

from .base_model import BasicClassifier

logger = logging.getLogger(__name__)

class LitModel(BasicClassifier):
    """
    A PyTorch Lightning model based on DenseNet121 for multi-label classification tasks.
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
        # Use the BasicClassifier constructor to set up the optimizer, etc.
        super().__init__(in_ch, out_ch, spatial_dims, optimizer=torch.optim.Adam, optimizer_kwargs={"lr": lr})

        self.lr = lr
        self.num_labels = num_labels
        self.seed = seed

        self.epoch_logs = []   # to track epoch logs
        self.current_train_loss = None
        self.current_lr = lr

        # For multi-label classification with BCE, use `task="multilabel"`.
        self.train_accuracy = Accuracy(task="multilabel", num_labels=num_labels)
        self.val_accuracy = Accuracy(task="multilabel", num_labels=num_labels)
        self.test_accuracy = Accuracy(task="multilabel", num_labels=num_labels)
        self.test_auroc = AUROC(task="multilabel", num_labels=num_labels)

        # Initialize the model
        if model_type == "densenet":
            self.model = densenet121(weights=DenseNet121_Weights.DEFAULT)
            num_ftrs = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_ftrs, out_ch)
            self.criterion = nn.BCEWithLogitsLoss()
        elif model_type == "resume":
            checkpoint = torch.load("results/checkpoint")
            self.model = checkpoint["model"]
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        # Define the loss function
        if criterion_name == "BCELoss":
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

        logger.info(
            f"Total Trainable Parameters: "
            f"{sum(p.numel() for p in self.model.parameters() if p.requires_grad)}"
        )

    def forward(self, x):
        """
        Forward pass for the model.
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        imgs, labels = batch[:2]
        preds = self(imgs)
        loss = self.criterion(preds, labels)

        acc = self.train_accuracy(preds, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_accuracy", acc, on_step=True, on_epoch=True, prog_bar=True)

        # You can store the latest train_loss if you want to log it manually
        self.current_train_loss = loss.detach()

        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch[:2]
        preds = self(imgs)
        loss = self.criterion(preds, labels)

        acc = self.val_accuracy(preds, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_accuracy", acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        imgs, labels, paths = batch
        preds = self(imgs)
        loss = self.criterion(preds, labels)

        acc = self.test_accuracy(preds, labels)
        auroc_val = self.test_auroc(preds, labels)

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_accuracy", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_auroc", auroc_val, on_step=False, on_epoch=True, prog_bar=True)

        # Optionally store predictions for analysis
        if not hasattr(self, "test_outputs"):
            self.test_outputs = []
        self.test_outputs.append({"preds": preds, "labels": labels, "paths": paths})
        return {"preds": preds, "labels": labels, "paths": paths}

    def configure_optimizers(self):
        """
        Configures the optimizer (and any LR schedulers if needed).
        """
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def on_train_epoch_end(self):
        """
        Called at the end of each training epoch.
        """
        optimizer = self.trainer.optimizers[0]
        self.current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"Epoch {self.current_epoch} completed. Current LR: {self.current_lr}")
        # Reset train accuracy for the next epoch
        self.train_accuracy.reset()

    def on_validation_epoch_start(self):
        # Reset val accuracy at the start of each val epoch
        self.val_accuracy.reset()

    def on_validation_epoch_end(self):
        """
        Called at the end of each validation epoch.
        """
        metrics = self.trainer.callback_metrics
        logger.info(f"[DEBUG] Callback Metrics at Validation Epoch End: {metrics}")

        val_loss = metrics.get("val_loss", None)
        val_accuracy = metrics.get("val_accuracy", None)

        logger.info(f"Validation metrics: Loss={val_loss}, Accuracy={val_accuracy}")

        epoch_log = {
            "epoch": self.current_epoch,
            "train_loss": (
                self.current_train_loss.item() if self.current_train_loss is not None else "N/A"
            ),
            "val_loss": val_loss.item() if val_loss is not None else "N/A",
            "val_accuracy": val_accuracy.item() if val_accuracy is not None else "N/A",
            "seed": self.seed,
            "lr": self.current_lr,
        }
        self.epoch_logs.append(epoch_log)
        pd.DataFrame(self.epoch_logs).to_csv("epoch_logs.csv", index=False)

    def on_test_epoch_start(self):
        # Reset test metrics
        self.test_accuracy.reset()
        self.test_auroc.reset()
        self.test_outputs = []

    def on_test_epoch_end(self):
        # Save test predictions if desired
        all_preds = torch.cat([x["preds"] for x in self.test_outputs], dim=0)
        all_labels = torch.cat([x["labels"] for x in self.test_outputs], dim=0)
        all_paths = sum([x["paths"] for x in self.test_outputs], [])

        torch.save({"preds": all_preds, "labels": all_labels, "paths": all_paths}, "test_predictions.pt")
        self.test_outputs = []  # Free memory
