
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torchmetrics.classification
import torchmetrics.segmentation
from torch import Tensor


class BaselineModule(pl.LightningModule):
    def __init__(self,
                 model: nn.Module,
                 loss: nn.Module,
                 class_weights: Tensor
        ):
        super(BaselineModule, self).__init__()
        # n classes
        n_classes = 5
        self.n_classes = n_classes

        self.class_weights = class_weights

        self.save_hyperparameters(ignore=['model', "loss"])

        self.model = model


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
        return self.model(x)

    def loss(self, images_pre: Tensor, masks_pre: Tensor, images_post: Tensor, masks_post: Tensor) -> Tensor:
        return self.loss_fn(self.forward(torch.cat([images_pre, images_post], dim=1)), masks_post.to(torch.float))  # type: ignore[no-any-return]

    def training_step(self, batch: list[Tensor], batch_idx: int) -> Tensor:
        images_pre, masks_pre, images_post, masks_post = batch
        preds = self.forward(torch.cat([images_pre, images_post], dim=1))
        loss = torch.stack([
            self.loss_fn(preds[:, i, ...], masks_post.to(torch.float)[:, i, ...]) for i in range(preds.shape[1])
        ]).dot(self.class_weights).sum()
        # todo log loss on each class
        self.log("train_loss", loss, prog_bar=True)
        return loss  # type: ignore[no-any-return]

    def validation_step(self, batch: list[Tensor], batch_idx: int):  # type: ignore[no-untyped-def]
        with torch.no_grad():
            images_pre, _, images_post, masks_post = batch

            cls_preds = self.forward(torch.cat([images_pre, images_post], dim=1))
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
                | {"val_loss": self.loss(*batch)}
            )
            self.log_dict(log_dict, prog_bar=True)

            return log_dict

    # todo proper params + scheduler
    def configure_optimizers(self):  # type: ignore[no-untyped-def]
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.0002, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            gamma=0.5,
            milestones=[5, 11, 17, 23, 29, 33, 47, 50, 60, 70, 90, 110, 130, 150, 170, 180, 190]
         )
        return [optimizer], [scheduler]
