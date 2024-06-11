import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl
import torchmetrics.classification
import torchmetrics.segmentation

from inz.models.unet_siamese import UNetSiamese


class SemanticSegmentorSiamese(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        localization_loss: nn.Module,
        classification_loss: nn.Module,
        n_classes: int,
    ):
        super(SemanticSegmentorSiamese, self).__init__()
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

        preds = self.model(images_pre, images_post)

        loc_y = masks_pre.argmax(dim=1).gt(0).to(torch.float)
        loc_loss = self.localization_loss(
            preds.argmax(dim=1).gt(0).to(torch.float),
            loc_y
        )

        cls_loss = self.classification_loss(preds, masks_post.argmax(dim=1))

        total_loss = loc_loss + cls_loss

        log_dict = {
            "loc_loss": loc_loss,
            "cls_loss": cls_loss,
            "loss": total_loss,
        }
        self.log_dict(log_dict, prog_bar=True)

        return log_dict

    def validation_step(self, batch: list[Tensor], batch_idx: int):  # type: ignore[no-untyped-def]
        with torch.no_grad():
            images_pre, _, images_post, masks_post = batch

            cls_preds = self.model(images_pre, images_post)
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


class DiceLoss(nn.Module):
    def __init__(self, weight=None, normalization="sigmoid"):  # type: ignore
        super(DiceLoss, self).__init__()
        self.register_buffer("weight", weight)
        assert normalization in ["sigmoid", "softmax", "none"]
        if normalization == "sigmoid":
            self.normalization = nn.Sigmoid()
        elif normalization == "softmax":
            self.normalization = nn.Softmax(dim=1)
        else:
            self.normalization = lambda x: x

    @staticmethod
    def flatten(tensor):  # type: ignore
        """Flattens a given tensor such that the channel axis is first.
        The shapes are transformed as follows:
        (N, C, D, H, W) -> (C, N * D * H * W)
        """
        # number of channels
        C = tensor.size(1)
        # new axis order
        axis_order = (1, 0) + tuple(range(2, tensor.dim()))
        # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
        transposed = tensor.permute(axis_order)
        # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
        return transposed.contiguous().view(C, -1)

    def dice(self, input, target, weight, epsilon=1e-6):  # type: ignore
        print(input.shape, target.shape)

        assert input.size() == target.size(), "'input' and 'target' must have the same shape"

        input = self.flatten(input)
        target = self.flatten(target)
        target = target.float()

        # compute per channel Dice Coefficient
        intersect = (input * target).sum(-1)
        if weight is not None:
            intersect = weight * intersect

        # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
        denominator = (input * input).sum(-1) + (target * target).sum(-1)
        return 2 * (intersect / denominator.clamp(min=epsilon))

    def forward(self, input, target):  # type: ignore
        # get probabilities from logits
        input = self.normalization(input)

        # compute per channel Dice coefficient
        per_channel_dice = self.dice(input, target, weight=self.weight)

        # average Dice score across all channels/classes
        return 1.0 - torch.mean(per_channel_dice)
