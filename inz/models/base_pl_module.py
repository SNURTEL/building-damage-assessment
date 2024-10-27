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

import gc


class BasePLModule(pl.LightningModule, ABC):
    """Base class for segmentation PL modules."""

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

        # region DELETE THIS IF PREVIOUS METRICS WERE TRULY BROKEN

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

        self.f1_loc = torchmetrics.classification.BinaryF1Score()

        # endregion

        # fixed metric computation
        self.accuracy_loc_safe = torchmetrics.classification.BinaryAccuracy()
        self.iou_loc_safe = torchmetrics.segmentation.MeanIoU(num_classes=2)

        self.f1_safe = torchmetrics.classification.MulticlassF1Score(num_classes=n_classes)
        self.precision_safe = torchmetrics.classification.MulticlassPrecision(num_classes=n_classes)
        self.recall_safe = torchmetrics.classification.MulticlassRecall(num_classes=n_classes)
        self.iou_safe = torchmetrics.segmentation.MeanIoU(num_classes=n_classes)

        self.f1_per_class_safe = torchmetrics.classification.MulticlassF1Score(num_classes=n_classes, average="none")
        self.precision_per_class_safe = torchmetrics.classification.MulticlassPrecision(
            num_classes=n_classes, average="none"
        )
        self.recall_per_class_safe = torchmetrics.classification.MulticlassRecall(num_classes=n_classes, average="none")
        self.iou_per_class_safe = torchmetrics.segmentation.MeanIoU(num_classes=n_classes, per_class=True)

        self.f1_loc_safe = torchmetrics.classification.BinaryF1Score()

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
            A tuple of two elements: 1) scalar loss value 2) per-class loss tensor of shape (5)
        """
        ...

    def training_step(self, batch: list[Tensor], batch_idx: int) -> Tensor:
        loss, class_loss = self.loss(*batch)

        class_loss_dict = {f"train_loss_{i}": loss_val for i, loss_val in enumerate(class_loss)}
        self.log_dict(class_loss_dict | {"train_loss": loss}, prog_bar=True, batch_size=batch[0].shape[0])
        return loss  # type: ignore[no-any-return]

    def on_train_epoch_start(self, *args, **kwargs):
        # lol
        # gc.collect()
        return super().on_train_epoch_start(*args, **kwargs)

    def validation_step(self, batch: list[Tensor], batch_idx: int):  # type: ignore[no-untyped-def]
        with torch.no_grad():
            images_pre, _, images_post, masks_post = batch

            cls_preds = self.forward(torch.cat([images_pre, images_post], dim=1))
            cls_preds_masks = F.one_hot(cls_preds.argmax(dim=1), num_classes=self.n_classes).moveaxis(-1, 1)

            cls_preds_argmax = cls_preds.argmax(dim=1)
            masks_post_argmax = masks_post.argmax(dim=1)

            loss, class_loss = self.loss(*batch)

            # region DELETE THIS IF PREVIOUS METRICS WERE TRULY BROKEN
            log_dict = (
                {
                    "acc_loc": self.accuracy_loc(
                        cls_preds_argmax.gt(0).to(torch.float), masks_post_argmax.gt(0).to(torch.float)
                    ),
                    "iou_loc": self.iou_loc(
                        F.one_hot(cls_preds_argmax.gt(0).to(torch.long), num_classes=2).moveaxis(-1, 1),
                        F.one_hot(masks_post_argmax.gt(0).to(torch.long), num_classes=2).moveaxis(-1, 1),
                    ),
                }
                | {
                    name: getattr(self, name)(cls_preds_argmax, masks_post_argmax)
                    for name in ["f1", "precision", "recall"]
                }
                | {"iou": self.iou(cls_preds_masks, masks_post.to(torch.uint8))}
                | {
                    f"{name}_{i}": val
                    for name, vec in {
                        name: getattr(self, f"{name}_per_class")(cls_preds_argmax, masks_post_argmax)
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
            log_dict = log_dict | {
                "f1_class": self.n_classes / sum([1 / v for k, v in log_dict.items() if k.startswith("f1_")]),
                "f1_loc": self.f1_loc(
                    (cls_preds_argmax > 0).to(torch.int), (masks_post_argmax > 0).to(torch.int)
                ),
            }
            log_dict = log_dict | {"challenge_score": 0.3 * log_dict["f1_loc"] + 0.7 * log_dict["f1_class"]}
            self.log_dict(log_dict, prog_bar=True, batch_size=batch[0].shape[0])

            # endregion

            self.accuracy_loc_safe(
                cls_preds_argmax.gt(0).to(torch.float), masks_post_argmax.gt(0).to(torch.float)
            )
            self.iou_loc_safe(
                F.one_hot(cls_preds_argmax.gt(0).to(torch.long), num_classes=2).moveaxis(-1, 1),
                F.one_hot(masks_post_argmax.gt(0).to(torch.long), num_classes=2).moveaxis(-1, 1),
            )
            self.f1_safe(cls_preds_argmax, masks_post_argmax)
            self.precision_safe(cls_preds_argmax, masks_post_argmax)
            self.recall_safe(cls_preds_argmax, masks_post_argmax)
            self.iou_safe(cls_preds_masks, masks_post.to(torch.uint8))
            self.f1_per_class_safe(cls_preds_argmax, masks_post_argmax)
            self.precision_per_class_safe(cls_preds_argmax, masks_post_argmax)
            self.recall_per_class_safe(cls_preds_argmax, masks_post_argmax)
            self.iou_per_class_safe(cls_preds_masks, masks_post.to(torch.uint8))
            self.f1_loc_safe((cls_preds_argmax > 0).to(torch.int), (masks_post_argmax > 0).to(torch.int))

            log_dict_safe = {
                "accuracy_loc_safe": self.accuracy_loc_safe,
                "iou_loc_safe": self.iou_loc_safe,
                "f1_safe": self.f1_safe,
                "precision_safe": self.precision_safe,
                "recall_safe": self.recall_safe,
                "iou_safe": self.iou_safe,
                "f1_loc_safe": self.f1_loc_safe,
            }

            self.log_dict(log_dict_safe, prog_bar=True, batch_size=batch[0].shape[0], on_epoch=True)
            for metric_name in ("f1", "precision", "recall", "iou"):
                for i in range(self.n_classes):
                    metric_attr = f"{metric_name}_per_class_safe"
                    self.log(
                        f"{metric_name}_{i}_safe",
                        getattr(self, metric_attr)[i],
                        metric_attribute=metric_attr,
                        prog_bar=True,
                        batch_size=batch[0].shape[0],
                        on_epoch=True,
                    )


    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_start()
        f1_per_class_safe = self.f1_per_class_safe.compute()
        f1_class_safe = self.n_classes / sum([1 / f1_per_class_safe[i] for i in range(self.n_classes)])
        challenge_score_safe = 0.3 * self.f1_loc_safe.compute() + 0.7 * f1_class_safe
        self.log("f1_class_safe", f1_class_safe, prog_bar=True)
        self.log("challenge_score_safe", challenge_score_safe, prog_bar=True)

    def test_step(self, *args, **kwargs) -> Tensor:
        return self.validation_step(*args, **kwargs)

    def on_test_epoch_end(self) -> None:
        super().on_validation_epoch_start()
        f1_per_class_safe = self.f1_per_class_safe.compute()
        f1_class_safe = self.n_classes / sum([1 / f1_per_class_safe[i] for i in range(self.n_classes)])
        challenge_score_safe = 0.3 * self.f1_loc_safe.compute() + 0.7 * f1_class_safe
        self.log("f1_class_safe", f1_class_safe, prog_bar=True)
        self.log("challenge_score_safe", challenge_score_safe, prog_bar=True)



    def configure_optimizers(self):  # type: ignore[no-untyped-def]
        optimizer = self.optimizer_factory(self.model.parameters())
        if self.scheduler_factory:
            scheduler = self.scheduler_factory(optimizer)
            return [optimizer], [scheduler]
        else:
            return optimizer
