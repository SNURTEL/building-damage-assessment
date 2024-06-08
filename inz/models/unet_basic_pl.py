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

        self.localization_loss = localization_loss
        self.classification_loss = classification_loss

        self.save_hyperparameters()

        self.f1 = torchmetrics.classification.MulticlassF1Score(num_classes=n_classes)
        self.precision = torchmetrics.classification.MulticlassPrecision(num_classes=n_classes)
        self.recall = torchmetrics.classification.MulticlassRecall(num_classes=n_classes)
        self.iou = torchmetrics.segmentation.MeanIoU(num_classes=n_classes)

    def training_step(self, batch: list[Tensor], batch_idx: int):  # type: ignore[no-untyped-def]
        images_pre, masks_pre, images_post, masks_post = batch

        loc_preds = self.model(images_pre)
        loc_y = masks_pre.argmax(dim=1).gt(0).to(torch.long)
        loc_loss = self.localization_loss(loc_y, loc_preds)

        cls_preds = self.model(images_post)
        cls_loss = self.classification_loss(masks_post.argmax(dim=1).gt(0).to(torch.long), cls_preds)

        total_loss = loc_loss + cls_loss

        log_dict = {
            "loc_loss": loc_loss,
            "cls_loss": cls_loss,
            "loss": total_loss
        }
        self.log_dict(log_dict)

        return log_dict

    def validation_step(self, batch: list[Tensor], batch_idx: int):  # type: ignore[no-untyped-def]
        with torch.no_grad():
            _, _, images_post, masks_post = batch

            cls_preds = self.model(images_post)


            f1 = self.f1(masks_post, cls_preds)

            log_dict = {
                "f1": f1
            }
            self.log_dict(log_dict)

            return log_dict

    def configure_optimizers(self):  # type: ignore[no-untyped-def]
        return torch.optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999))


class OrdinalCrossEntropyLoss(nn.Module):
    def __init__(self, n_classes: int, weights: Tensor | None = None, reduction: str = "mean"):
        super(OrdinalCrossEntropyLoss, self).__init__()
        self.n_classes = n_classes
        self.weights = weights or torch.ones(n_classes).cuda()
        self.reduction = reduction

    def forward(self, y: Tensor, y_hat: Tensor) -> Tensor:
        ce_loss = F.cross_entropy(y_hat, y, reduction=self.reduction, weight=self.weights)
        distance_weight = torch.abs(y_hat.argmax(1) - y) + 1
        return torch.mean(distance_weight * ce_loss)
