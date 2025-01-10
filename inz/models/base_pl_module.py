import io
from abc import ABC, abstractmethod

import pytorch_lightning as pl
import pytorch_lightning.loggers
import torch
import torch.nn.functional as F
import torchmetrics
import torchmetrics.classification
import torchmetrics.segmentation
import torchvision
from matplotlib import pyplot as plt
from PIL import Image
from torch import Tensor


class BasePLModule(pl.LightningModule, ABC):
    """Base class for segmentation PL modules."""

    def __init__(
        self,
        model,
        optimizer_factory,
        scheduler_factory=None,
        class_weights=None,
        n_classes=5,
    ):
        """
        Initializes the BasePLModule.

        Args:
            model: The neural network model.
            optimizer_factory: A callable that creates an optimizer.
            scheduler_factory: A callable that creates a learning rate scheduler (optional).
            class_weights: The weights for each class (optional).
            n_classes: The number of classes (default is 5).
        """
        super(BasePLModule, self).__init__()
        # n classes
        self.n_classes = n_classes

        self.class_weights = class_weights

        self.save_hyperparameters(ignore=["model", "loss"])

        self.model = model

        self.optimizer_factory = optimizer_factory
        self.scheduler_factory = scheduler_factory

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

        self.confusion_matrix_safe = torchmetrics.classification.MulticlassConfusionMatrix(
            num_classes=n_classes, normalize="true"
        )

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagate the input through the network.

        Args:
            x: An input tensor of shape (N, 6, H, W)

        Returns:
            An output tensor of shape (N, C, H, W)
        """
        ...

    @abstractmethod
    def loss(
        self, images_pre: Tensor, masks_pre: Tensor, images_post: Tensor, masks_post: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Compute the global and class-wise loss.

        Args:
            images_pre: Images pre-disaster [-1, 1], (N, 3, H, W)
            masks_pre: Label masks pre-disaster {0, 1}, (N, C, H, W)
            images_post: Images post-disaster [-1, 1], (N, 3, H, W)
            masks_post: Label masks post-disaster {0, 1}, (N, C, H, W)

        Returns:
            A tuple of two elements: 1) scalar loss value 2) per-class loss tensor of shape (C)
        """
        ...

    def training_step(self, batch: list[Tensor], batch_idx: int) -> Tensor:
        """
        Perform a single training step.

        Args:
            batch: A list of tensors representing the input batch.
            batch_idx: An integer representing the index of the current batch.

        Returns:
            The loss value for the training step.
        """
        loss, class_loss = self.loss(*batch)

        class_loss_dict = {f"train_loss_{i}": loss_val for i, loss_val in enumerate(class_loss)}
        self.log_dict(class_loss_dict | {"train_loss": loss}, prog_bar=True, batch_size=batch[0].shape[0])
        return loss  # type: ignore[no-any-return]

    def validation_step(self, batch, batch_idx):
        """
        Perform a validation step on the given batch of data.

        Args:
            batch: A list containing the input tensors for the batch.
            batch_idx: An integer representing the index of the current batch.

        Returns:
            None
        """
        with torch.no_grad():
            images_pre, _, images_post, masks_post = batch

            cls_preds = self(torch.cat([images_pre, images_post], dim=1))
            cls_preds_masks = F.one_hot(cls_preds.argmax(dim=1), num_classes=self.n_classes).moveaxis(-1, 1)

            cls_preds_argmax = cls_preds.argmax(dim=1)
            masks_post_argmax = masks_post.argmax(dim=1)

            loss, class_loss = self.loss(*batch)

            self.accuracy_loc_safe(cls_preds_argmax.gt(0).to(torch.float), masks_post_argmax.gt(0).to(torch.float))
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
            self.confusion_matrix_safe(cls_preds_argmax, masks_post_argmax)

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
        """
        Hook function called at the end of the validation epoch.

        Computes the f1_per_class_safe, f1_class_safe, and challenge_score_safe metrics
        and logs them using the LightningLoggerBase.

        """
        super().on_validation_epoch_end()
        f1_per_class_safe = self.f1_per_class_safe.compute()
        f1_class_safe = self.n_classes / sum([1 / f1_per_class_safe[i] for i in range(self.n_classes)])
        challenge_score_safe = 0.3 * self.f1_loc_safe.compute() + 0.7 * f1_class_safe
        self.log("f1_class_safe", f1_class_safe, prog_bar=True)
        self.log("challenge_score_safe", challenge_score_safe, prog_bar=True)

    def test_step(self, *args, **kwargs) -> Tensor:
        """
        Perform a single step of testing (equal to validation step).

        Returns:
            The output tensor of the testing step.
        """
        self.eval()
        return self.validation_step(*args, **kwargs)

    def on_test_epoch_end(self) -> None:
        """
        Hook method called at the end of the testing epoch.

        This method calculates the f1_class_safe and challenge_score_safe metrics,
        logs them using the logger, and saves the confusion matrix as an image.

        Returns:
            None
        """
        super().on_test_epoch_end()
        f1_per_class_safe = self.f1_per_class_safe.compute()
        f1_class_safe = self.n_classes / sum([1 / f1_per_class_safe[i] for i in range(self.n_classes)])
        challenge_score_safe = 0.3 * self.f1_loc_safe.compute() + 0.7 * f1_class_safe
        self.log("f1_class_safe", f1_class_safe, prog_bar=True)
        self.log("challenge_score_safe", challenge_score_safe, prog_bar=True)

        if not hasattr(self.logger, "log_image"):
            return

        fig, ax = plt.subplots(figsize=(10, 10))

        self.confusion_matrix_safe.plot(
            ax=ax, labels=["Background", "No damage", "Minor damage", "Major damage", "Destroyed"], cmap="Greens"
        )

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        im = torchvision.transforms.ToTensor()(Image.open(buf))
        self.logger.log_image(
            key="confusion_matrix",
            images=[im],
        )

    def configure_optimizers(self):  # type: ignore[no-untyped-def]
        """
        Configures the optimizers for the model.

        Returns:
            The configured optimizer and scheduler (is scheduler factory was specified).
        """
        optimizer = self.optimizer_factory(self.model.parameters())
        if self.scheduler_factory:
            scheduler = self.scheduler_factory(optimizer)
            return [optimizer], [scheduler]
        else:
            return optimizer
