from typing import Any, Callable

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torchmetrics.classification
import torchmetrics.segmentation
from torch import Tensor


class BaselineModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        loss: nn.Module,
        optimizer_factory: Callable[[Any], torch.optim.Optimizer],
        scheduler_factory: Callable[[Any], torch.optim.lr_scheduler.LRScheduler] | None = None,
        class_weights: Tensor | None = None,
    ):
        super(BaselineModule, self).__init__()
        # n classes
        n_classes = 5
        self.n_classes = n_classes

        self.class_weights = class_weights

        self.save_hyperparameters(ignore=["model", "loss"])

        self.model = model

        self.optimizer_factory = optimizer_factory
        self.scheduler_factory = scheduler_factory

        # loss function
        self.loss_fn = loss
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)  # type: ignore[no-any-return]

    def loss(
        self, images_pre: Tensor, masks_pre: Tensor, images_post: Tensor, masks_post: Tensor
    ) -> tuple[Tensor, Tensor]:
        preds = self.forward(torch.cat([images_pre, images_post], dim=1))
        if self.class_weights is not None:
            # per-class loss in unweighted!
            class_loss = torch.stack(
                [self.loss_fn(preds[:, i, ...], masks_post.to(torch.float)[:, i, ...]) for i in range(preds.shape[1])]
            )
            loss = class_loss.dot(self.class_weights).sum()
        else:
            loss = self.loss_fn(preds, masks_post.to(torch.float))
            class_loss = Tensor([0, 0, 0, 0, 0])

        return loss, class_loss

    def training_step(self, batch: list[Tensor], batch_idx: int) -> Tensor:
        loss, class_loss = self.loss(*batch)

        class_loss_dict = {f"train_loss_{i}": loss_val for i, loss_val in enumerate(class_loss)}
        self.log_dict(class_loss_dict | {"train_loss": loss}, prog_bar=True, batch_size=batch.shape[0])
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
            self.log_dict(log_dict, prog_bar=True, batch_size=batch.shape[0])

            return log_dict

    def configure_optimizers(self):  # type: ignore[no-untyped-def]
        optimizer = self.optimizer_factory(self.model.parameters())
        if self.scheduler_factory:
            scheduler = self.scheduler_factory(optimizer)
            return [optimizer], [scheduler]
        else:
            return optimizer
