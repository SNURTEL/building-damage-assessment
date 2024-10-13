from abc import ABC, abstractmethod
from typing import Any, Callable

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torchmetrics.classification
import torchmetrics.segmentation
from torch import Tensor


class BasePLModule(pl.LightningModule, ABC):
    """Base class for segmentation PL modules.
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer_factory: Callable[[Any], torch.optim.Optimizer],
        scheduler_factory: Callable[[Any], torch.optim.lr_scheduler.LRScheduler] | None = None,
        class_weights: Tensor | None = None,
    ):
        super(BasePLModule, self).__init__()
        # n classes
        n_classes = 5
        self.n_classes = n_classes

        self.class_weights = class_weights

        self.save_hyperparameters(ignore=["model", "loss"])

        self.model = model

        self.optimizer_factory = optimizer_factory
        self.scheduler_factory = scheduler_factory

        # metrics
        self.accuracy_loc = torchmetrics.classification.BinaryAccuracy()
        self.iou_loc = torchmetrics.segmentation.MeanIoU(num_classes=2)

        self.f1 = torchmetrics.classification.MulticlassF1Score(num_classes=n_classes)
        self.precision = torchmetrics.classification.MulticlassPrecision(num_classes=n_classes)
        self.recall = torchmetrics.classification.MulticlassRecall(num_classes=n_classes)
        self.iou = torchmetrics.segmentation.MeanIoU(num_classes=n_classes)

        self.f1_per_class = torchmetrics.classification.MulticlassF1Score(num_classes=n_classes, average="none")
        self.precision_per_class = torchmetrics.classification.MulticlassPrecision(
            num_classes=n_classes, average="none"
        )
        self.recall_per_class = torchmetrics.classification.MulticlassRecall(num_classes=n_classes, average="none")
        self.iou_per_class = torchmetrics.segmentation.MeanIoU(num_classes=n_classes, per_class=True)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagate the input through the network.

        Args:
            x: An input tensor of shape (N, 6, H, W)

        Returns:
            An output tensor of shape (N, 5, H, W)
        """
        ...

    @abstractmethod
    def loss(
        self, images_pre: Tensor, masks_pre: Tensor, images_post: Tensor, masks_post: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Compute the global and class-wise loss.

        Args:
            images_pre: Images pre-disaster [-1, 1], (N, 3, H, W)
            masks_pre: Label masks pre-disaster {0, 1}, (N, 5, H, W)
            images_post: Images post-disaster [-1, 1], (N, 3, H, W)
            masks_post: Label masks post-disaster {0, 1}, (N, 5, H, W)

        Returns:
            _description_
        """
        ...

    def training_step(self, batch: list[Tensor], batch_idx: int) -> Tensor:
        loss, class_loss = self.loss(*batch)

        class_loss_dict = {f"train_loss_{i}": loss_val for i, loss_val in enumerate(class_loss)}
        self.log_dict(class_loss_dict | {"train_loss": loss}, prog_bar=True, batch_size=batch[0].shape[0])
        return loss  # type: ignore[no-any-return]

    def validation_step(self, batch: list[Tensor], batch_idx: int):  # type: ignore[no-untyped-def]
        with torch.no_grad():
            images_pre, _, images_post, masks_post = batch

            cls_preds = self.forward(torch.cat([images_pre, images_post], dim=1))
            cls_preds_masks = F.one_hot(cls_preds.argmax(dim=1), num_classes=self.n_classes).moveaxis(-1, 1)

            loss, class_loss = self.loss(*batch)

            log_dict = (
                {
                    "acc_loc": self.accuracy_loc(
                        cls_preds.argmax(dim=1).gt(0).to(torch.float), masks_post.argmax(dim=1).gt(0).to(torch.float)
                    ),
                    "iou_loc": self.iou_loc(
                        F.one_hot(cls_preds.argmax(dim=1).gt(0).to(torch.long), num_classes=2).moveaxis(-1, 1),
                        F.one_hot(masks_post.argmax(dim=1).gt(0).to(torch.long), num_classes=2).moveaxis(-1, 1),
                    ),
                }
                | {
                    name: getattr(self, name)(cls_preds.argmax(dim=1), masks_post.argmax(dim=1))
                    for name in ["f1", "precision", "recall"]
                }
                | {"iou": self.iou(cls_preds_masks, masks_post.to(torch.uint8))}
                | {
                    f"{name}_{i}": val
                    for name, vec in {
                        name: getattr(self, f"{name}_per_class")(cls_preds.argmax(dim=1), masks_post.argmax(dim=1))
                        for name in ["f1", "precision", "recall"]
                    }.items()
                    for i, val in enumerate(vec)
                }
                | {
                    f"iou_{i}": val
                    for i, val in enumerate(self.iou_per_class(cls_preds_masks, masks_post.to(torch.uint8)))
                }
                | {f"val_loss_{i}": loss_val for i, loss_val in enumerate(class_loss)}
                | {"val_loss": loss}
            )
            self.log_dict(log_dict, prog_bar=True, batch_size=batch[0].shape[0])

            return log_dict

    def configure_optimizers(self):  # type: ignore[no-untyped-def]
        optimizer = self.optimizer_factory(self.model.parameters())
        if self.scheduler_factory:
            scheduler = self.scheduler_factory(optimizer)
            return [optimizer], [scheduler]
        else:
            return optimizer
