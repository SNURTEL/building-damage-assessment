import pytorch_lightning as pl
import segmentation_models_pytorch as smp  # type: ignore[import-untyped]
import torch
import torch.nn as nn
import torchmetrics
import torchmetrics.classification
import torchmetrics.segmentation
from torch import Tensor
from torch.nn import functional as F


class ConvReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super(ConvReLU, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1),
            nn.ReLU(
                inplace=True,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x)  # type: ignore[no-any-return]


class UpConv(nn.Module):
    def __init__(self, in_channels: int, in_res_channels: int, out_channels: int):
        super(UpConv, self).__init__()
        self.conv1 = ConvReLU(in_channels=in_channels, out_channels=out_channels)
        self.conv2 = ConvReLU(in_channels=out_channels + in_res_channels, out_channels=out_channels)

    def forward(self, x: Tensor, x_res: Tensor) -> Tensor:
        x = self.conv1(F.interpolate(x, scale_factor=2))
        x = self.conv2(torch.cat([x, x_res], 1))
        return x


class BDANetSimplified(pl.LightningModule):
    def __init__(
        self,
        n_classes: int,
        loss: nn.Module,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
    ):
        super(BDANetSimplified, self).__init__()

        # hyperparams
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2

        self.save_hyperparameters()

        # n classes
        self.n_classes = n_classes

        # loss function
        self.loss_fn = loss

        # model structure
        encoder = smp.encoders.get_encoder(name="se_resnext50_32x4d", in_channels=3, weights="imagenet")

        # filter sizes for se_resnext50_32x4d
        enc_filters = [64, 256, 512, 1024, 2048]
        dec_filters = [32, 48, 64, 128, 256]

        # taken straight from source code; unfortunately, we need to access individual layers directly
        # to extract latent representations at each layer
        self.down_conv1 = encoder.layer0[:-1]
        self.down_conv2 = nn.Sequential(encoder.layer0[-1], encoder.layer1)
        self.down_conv3 = encoder.layer2
        self.down_conv4 = encoder.layer3
        self.down_conv5 = encoder.layer4

        self.up_conv54 = UpConv(
            in_channels=enc_filters[-1], in_res_channels=enc_filters[-2], out_channels=dec_filters[-1]
        )
        self.up_conv43 = UpConv(
            in_channels=dec_filters[-1], in_res_channels=enc_filters[-3], out_channels=dec_filters[-2]
        )
        self.up_conv32 = UpConv(
            in_channels=dec_filters[-2], in_res_channels=enc_filters[-4], out_channels=dec_filters[-3]
        )
        self.up_conv21 = UpConv(
            in_channels=dec_filters[-3], in_res_channels=enc_filters[-5], out_channels=dec_filters[-4]
        )
        self.up_conv1_nores = ConvReLU(dec_filters[-4], dec_filters[-5])

        self.outconv = nn.Conv2d(
            in_channels=dec_filters[-5] * 2, out_channels=n_classes, kernel_size=1, stride=1, padding=0
        )

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

    def _forward_single(self, x: Tensor) -> Tensor:
        enc1 = self.down_conv1(x)
        enc2 = self.down_conv2(enc1)
        enc3 = self.down_conv3(enc2)
        enc4 = self.down_conv4(enc3)
        enc5 = self.down_conv5(enc4)

        dec4 = self.up_conv54(enc5, enc4)
        dec3 = self.up_conv43(dec4, enc3)
        dec2 = self.up_conv32(dec3, enc2)
        dec1 = self.up_conv21(dec2, enc1)

        dec0 = self.up_conv1_nores(F.interpolate(dec1, scale_factor=2))

        return dec0  # type: ignore[no-any-return]

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x1 = self._forward_single(x1)
        x2 = self._forward_single(x2)
        return self.outconv(torch.cat([x1, x2], 1))  # type: ignore[no-any-return]

    def loss(self, images_pre: Tensor, masks_pre: Tensor, images_post: Tensor, masks_post: Tensor) -> Tensor:
        return self.loss_fn(self(images_pre, images_post), masks_post.to(torch.float))  # type: ignore[no-any-return]

    def training_step(self, batch: list[Tensor], batch_idx: int) -> Tensor:
        loss = self.loss(*batch)

        self.log("train_loss", loss, prog_bar=True)
        return loss  # type: ignore[no-any-return]

    def validation_step(self, batch: list[Tensor], batch_idx: int):  # type: ignore[no-untyped-def]
        with torch.no_grad():
            images_pre, _, images_post, masks_post = batch

            cls_preds = self(images_pre, images_post)
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

    def configure_optimizers(self):  # type: ignore[no-untyped-def]
        return torch.optim.Adam(self.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
