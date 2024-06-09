import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl
import torchmetrics.classification
import torchmetrics.segmentation

from inz.models.unet_basic import UNet


class SemanticSegmentor(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        localization_loss: nn.Module,
        classification_loss: nn.Module,
        n_classes: int,
    ):
        super(SemanticSegmentor, self).__init__()
        self.model = model
        self.n_classes = n_classes

        self.localization_loss = localization_loss
        self.classification_loss = classification_loss

        self.save_hyperparameters()

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

    def training_step(self, batch: list[Tensor], batch_idx: int):  # type: ignore[no-untyped-def]
        images_pre, masks_pre, images_post, masks_post = batch

        loc_preds = self.model(images_pre)
        loc_y = masks_pre.argmax(dim=1).gt(0).to(torch.float)
        loc_loss = self.localization_loss(
            loc_preds.argmax(dim=1).gt(0).to(torch.float), 
            loc_y
        )

        cls_preds = self.model(images_post)
        cls_loss = self.classification_loss(cls_preds, masks_post.argmax(dim=1).gt(0).to(torch.long))

        # total_loss = loc_loss + cls_loss
        total_loss = cls_loss

        log_dict = {"loc_loss": loc_loss, "cls_loss": cls_loss, "loss": total_loss}
        self.log_dict(log_dict, prog_bar=True)

        return log_dict

    def validation_step(self, batch: list[Tensor], batch_idx: int):  # type: ignore[no-untyped-def]
        with torch.no_grad():
            _, _, images_post, masks_post = batch

            cls_preds = self.model(images_post)
            cls_preds_masks = F.one_hot(cls_preds.argmax(dim=1), num_classes=self.n_classes).moveaxis(-1, 1)
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
            )
            self.log_dict(log_dict, prog_bar=True)

            return log_dict

    def configure_optimizers(self):  # type: ignore[no-untyped-def]
        return torch.optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999))


class OrdinalCrossEntropyLoss(nn.Module):
    def __init__(self, n_classes: int, weights: Tensor | None = None, reduction: str = "mean"):
        super(OrdinalCrossEntropyLoss, self).__init__()
        self.n_classes = n_classes
        self.weights = weights if weights is not None else torch.ones(n_classes).cuda()
        self.reduction = reduction

    def forward(self, y_hat: Tensor, y: Tensor) -> Tensor:
        ce_loss = F.cross_entropy(y_hat, y, reduction=self.reduction, weight=self.weights)
        distance_weight = torch.abs(y_hat.argmax(1) - y) + 1
        return torch.mean(distance_weight * ce_loss)
