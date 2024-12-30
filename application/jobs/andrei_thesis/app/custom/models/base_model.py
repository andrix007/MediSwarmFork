import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import List, Union, Dict
from torchmetrics import AUROC, Accuracy


class VeryBasicModel(pl.LightningModule):
    """
    A very basic model class extending LightningModule with minimal functionality.
    """

    def __init__(self):
        super().__init__()
        self._step_train = -1
        self._step_val = -1
        self._step_test = -1

    def forward(self, x):
        """Forward pass."""
        raise NotImplementedError("forward() must be implemented by subclasses")

    def _step(self, batch, batch_idx, state, step, optimizer_idx):
        """Shared step logic for train/val/test."""
        raise NotImplementedError("Step logic must be implemented by subclasses")

    def _epoch_end(self, outputs, state):
        """Epoch end logic."""
        return

    def training_step(self, batch, batch_idx):
        self._step_train += 1
        return self._step(batch, batch_idx, "train", self._step_train, 0)

    def validation_step(self, batch, batch_idx):
        self._step_val += 1
        return self._step(batch, batch_idx, "val", self._step_val, 0)

    def test_step(self, batch, batch_idx):
        self._step_test += 1
        return self._step(batch, batch_idx, "test", self._step_test, 0)

    def training_epoch_end(self, outputs):
        self._epoch_end(outputs, "train")

    def validation_epoch_end(self, outputs):
        self._epoch_end(outputs, "val")

    def test_epoch_end(self, outputs):
        self._epoch_end(outputs, "test")


class BasicModel(VeryBasicModel):
    """
    Base model class extending VeryBasicModel for federated learning.
    Includes optimizer, scheduler, metrics, and loss configuration.
    """

    def __init__(
        self,
        optimizer=torch.optim.Adam,
        optimizer_kwargs=None,
        lr_scheduler=None,
        lr_scheduler_kwargs=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs or {"lr": 1e-3}
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs or {}

        # Metrics
        self.train_metrics = nn.ModuleDict({
            "acc": Accuracy(task="multiclass", num_classes=8),  # Update task type based on your use case
            "auc_roc": AUROC(task="multiclass", num_classes=8)
        })
        self.val_metrics = nn.ModuleDict({
            "acc": Accuracy(task="multiclass", num_classes=8),
            "auc_roc": AUROC(task="multiclass", num_classes=8)
        })
        self.test_metrics = nn.ModuleDict({
            "acc": Accuracy(task="multiclass", num_classes=8),
            "auc_roc": AUROC(task="multiclass", num_classes=8)
        })

        # Loss
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        """Override this in subclasses."""
        raise NotImplementedError("forward() must be implemented by subclasses")

    def _shared_step(self, batch, state):
        """Shared logic for all steps."""
        inputs, targets = batch
        outputs = self(inputs)

        # Compute loss
        loss = self.loss_fn(outputs, targets)

        # Log metrics
        metrics = getattr(self, f"{state}_metrics")
        metrics["acc"].update(outputs, targets)
        metrics["auc_roc"].update(outputs, targets)

        self.log(f"{state}_loss", loss, on_step=True, on_epoch=True)
        for metric_name, metric in metrics.items():
            self.log(f"{state}_{metric_name}", metric.compute(), on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        optimizer = self.optimizer(self.parameters(), **self.optimizer_kwargs)
        if self.lr_scheduler:
            scheduler = self.lr_scheduler(optimizer, **self.lr_scheduler_kwargs)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return optimizer

    def on_train_epoch_end(self):
        """Reset metrics at the end of training."""
        for metric in self.train_metrics.values():
            metric.reset()

    def on_validation_epoch_end(self):
        """Reset metrics at the end of validation."""
        for metric in self.val_metrics.values():
            metric.reset()

    def on_test_epoch_end(self):
        """Reset metrics at the end of testing."""
        for metric in self.test_metrics.values():
            metric.reset()

class BasicClassifier(pl.LightningModule):
    """
    A basic classifier model for PyTorch Lightning.
    Includes functionality for federated learning compatibility with NVFlare.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        spatial_dims: int,
        loss=torch.nn.CrossEntropyLoss,
        loss_kwargs=None,
        optimizer=torch.optim.Adam,
        optimizer_kwargs=None,
        lr_scheduler=None,
        lr_scheduler_kwargs=None,
        aucroc_kwargs=None,
        acc_kwargs=None,
    ):
        """
        Initialize the basic classifier.

        Args:
            in_ch (int): Number of input channels.
            out_ch (int): Number of output channels.
            spatial_dims (int): Number of spatial dimensions for the input data.
            loss (callable, optional): Loss function. Defaults to CrossEntropyLoss.
            loss_kwargs (dict, optional): Arguments for the loss function.
            optimizer (callable, optional): Optimizer class. Defaults to Adam.
            optimizer_kwargs (dict, optional): Arguments for the optimizer.
            lr_scheduler (callable, optional): Learning rate scheduler class. Defaults to None.
            lr_scheduler_kwargs (dict, optional): Arguments for the learning rate scheduler.
            aucroc_kwargs (dict, optional): Arguments for the AUROC metric.
            acc_kwargs (dict, optional): Arguments for the Accuracy metric.
        """
        super().__init__()
        self.save_hyperparameters()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.spatial_dims = spatial_dims

        # Define loss function
        self.loss_fn = loss(**(loss_kwargs or {}))

        # Define optimizer and scheduler
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs or {"lr": 1e-3}
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs or {}

        # Define metrics
        aucroc_kwargs = aucroc_kwargs or {"task": "binary"}
        acc_kwargs = acc_kwargs or {"task": "binary"}
        self.auc_roc = nn.ModuleDict({state: AUROC(**aucroc_kwargs) for state in ["train_", "val_", "test_"]})
        self.acc = nn.ModuleDict({state: Accuracy(**acc_kwargs) for state in ["train_", "val_", "test_"]})

        # Define the model architecture
        self.model = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(32 * (spatial_dims // 2) ** 2, out_ch),  # Adjust dimensions as needed
        )

    def forward(self, x):
        """
        Forward pass through the model.
        Applies activation depending on the loss function.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.model(x)
        if isinstance(self.loss_fn, nn.CrossEntropyLoss):
            x = torch.softmax(x, dim=1)
        elif isinstance(self.loss_fn, nn.BCELoss):
            x = torch.sigmoid(x)
        return x

    def _step(self, batch: dict, batch_idx: int, state: str, step: int, optimizer_idx: int = 0):
        """
        Shared logic for training, validation, and testing steps.

        Args:
            batch (dict): Input batch containing 'source' and 'target'.
            batch_idx (int): Batch index.
            state (str): One of "train", "val", or "test".
            step (int): Current step.
            optimizer_idx (int): Index of the optimizer.

        Returns:
            torch.Tensor: Loss value for the step.
        """
        source, target = batch["source"], batch["target"]
        if target.ndim == 1:  # For binary classification, ensure targets have the correct shape
            target = target[:, None].float()

        batch_size = source.shape[0]
        pred = self(source)  # Forward pass
        loss = self.loss_fn(pred, target)  # Compute loss

        # Update metrics
        self.acc[state + "_"].update(pred, target)
        self.auc_roc[state + "_"].update(pred, target)

        # Log metrics and loss
        self.log(f"{state}_loss", loss, batch_size=batch_size, on_step=True, on_epoch=True)
        self.log(f"{state}_accuracy", self.acc[state + "_"].compute(), batch_size=batch_size, on_step=False, on_epoch=True)
        self.log(f"{state}_auc_roc", self.auc_roc[state + "_"].compute(), batch_size=batch_size, on_step=False, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train", self.global_step)

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val", self.global_step)

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "test", self.global_step)

    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers.

        Returns:
            dict: Dictionary containing the optimizer and optionally the scheduler.
        """
        optimizer = self.optimizer(self.parameters(), **self.optimizer_kwargs)
        if self.lr_scheduler is not None:
            scheduler = self.lr_scheduler(optimizer, **self.lr_scheduler_kwargs)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return optimizer

    def _reset_metrics(self, state: str):
        """
        Reset metrics for the specified state.

        Args:
            state (str): One of "train", "val", or "test".
        """
        self.acc[state + "_"].reset()
        self.auc_roc[state + "_"].reset()

    def training_epoch_end(self, outputs):
        self._reset_metrics("train_")

    def validation_epoch_end(self, outputs):
        self._reset_metrics("val_")

    def test_epoch_end(self, outputs):
        self._reset_metrics("test_")

    # NVFlare compatibility methods
    def get_state(self) -> Dict:
        """
        Retrieve the model state for federated learning.

        Returns:
            dict: Dictionary containing the model state.
        """
        return {"model": self.state_dict()}

    def set_state(self, state: Dict):
        """
        Load the model state for federated learning.

        Args:
            state (dict): Dictionary containing the model state.
        """
        self.load_state_dict(state["model"])
