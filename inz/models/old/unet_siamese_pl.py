import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torchmetrics.classification
import torchmetrics.segmentation
from torch import Tensor

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

    def loss(self, images_pre: Tensor, masks_pre: Tensor, images_post: Tensor, masks_post: Tensor) -> Tensor:
        preds = self.model(images_pre, images_post)

        # loc_y = masks_pre.argmax(dim=1).gt(0).to(torch.float)
        # loc_loss = self.localization_loss(preds.argmax(dim=1).gt(0).to(torch.float), loc_y)

        # for ordinal cross entropy, TODO unify interfaces
        # cls_loss = self.classification_loss(preds, masks_post.argmax(dim=1))

        cls_loss = self.classification_loss(
            preds, masks_post.to(torch.float)
        )

        # total_loss = loc_loss + cls_loss
        total_loss = cls_loss

        return {
            # "loc_loss": loc_loss,
            "cls_loss": cls_loss,
            "loss": total_loss,
        }

    def training_step(self, batch: list[Tensor], batch_idx: int):  # type: ignore[no-untyped-def]
        log_dict = self.loss(*batch)

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
                } | {f"val_{k}": v for k, v in self.loss(*batch).items()}
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
    def __init__(self, weights=None, normalization="sigmoid"):  # type: ignore
        super(DiceLoss, self).__init__()
        self.register_buffer("weights", weights)
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

    def dice(self, input, target, weights, epsilon=1e-6):  # type: ignore
        assert input.size() == target.size(), "'input' and 'target' must have the same shape"

        input = self.flatten(input)
        target = self.flatten(target)
        target = target.float()

        # compute per channel Dice Coefficient
        intersect = (input * target).sum(-1)
        if weights is not None:
            intersect = weights * intersect

        # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
        denominator = (input * input).sum(-1) + (target * target).sum(-1)
        return 2 * (intersect / denominator.clamp(min=epsilon))

    def forward(self, input, target):  # type: ignore
        # get probabilities from logits
        input = self.normalization(input)

        # compute per channel Dice coefficient
        per_channel_dice = self.dice(input, target, weights=self.weights)

        # average Dice score across all channels/classes
        return 1.0 - torch.mean(per_channel_dice)


class CrossEntropyDiceLoss(nn.Module):
    def __init__(self, weights: Tensor | None =None, reduction: str="mean") -> None:
        super(CrossEntropyDiceLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(weight=weights, reduction=reduction)
        self.dice = DiceLoss(weights=weights)

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        return self.ce(inputs, targets) + self.dice(inputs, targets)


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2, reduction: str = "mean", weight: Tensor | None = None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        if reduction == "none":
            self.reduction_fn = lambda t: t
        elif reduction == "mean":
            self.reduction_fn = lambda t: t.mean()
        elif reduction == "sum":
            self.reduction_fn = lambda t: t.sum()
        else:
            raise ValueError(
                f"Invalid Value for arg 'reduction': '{self.reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
            )
        self.reduction = reduction
        self.weight = weight

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none", pos_weight=self.weight if self.weight is not None else torch.tensor(1)).cuda()
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        return self.reduction_fn(loss)
