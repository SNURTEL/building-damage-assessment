import io
from abc import abstractmethod
from typing import Any, Callable

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
import torchmetrics.classification
import torchmetrics.segmentation
import torchvision
from matplotlib import pyplot as plt
from PIL import Image
from torch import Tensor

from inz.models.base_pl_module import BasePLModule


class BaseMSLModuleWrapper(pl.LightningModule):
    def __init__(
        self,
        pl_module: BasePLModule,
        n_classes_target: int,
        msl_loss_module: torch.nn.Module,
        msl_lambda: float,
        optimizer_factory: Callable[[Any], torch.optim.Optimizer],
        scheduler_factory: Callable[[Any], torch.optim.lr_scheduler.LRScheduler] | None = None,
        target_conf_matrix_labels: list[str] | None = None,
    ):
        """
        Initializes the MSLModuleWrapper.

        Args:
            pl_module (BasePLModule): The base PyTorch Lightning module.
            n_classes_target (int): The number of target classes.
            msl_loss_module (torch.nn.Module): The MSL loss module.
            msl_lambda (float): The MSL lambda param value.
            optimizer_factory (Callable[[Any], torch.optim.Optimizer]): The optimizer factory function.
            scheduler_factory (Callable[[Any], torch.optim.lr_scheduler.LRScheduler] | None, optional):
                The LR scheduler factory function. Defaults to None.
            target_conf_matrix_labels (list[str] | None, optional): The target confusion matrix labels.
                Defaults to None.
        """
        super(BaseMSLModuleWrapper, self).__init__()
        n_classes_source = pl_module.n_classes
        self.n_classes_source = n_classes_source
        self.n_classes_target = n_classes_target

        self.save_hyperparameters(ignore=["pl_module", "msl_loss_module", "target_conf_matrix_labels"])

        self.automatic_optimization = False

        self.inner = pl_module

        # as the inner PL module is not managed by Trainer, log attempts would crash the training
        self.inner.log = self.log
        self.inner.log_dict = self.log_dict

        self.msl_loss = msl_loss_module
        self.msl_lambda = msl_lambda

        self.optimizer_factory = optimizer_factory
        self.scheduler_factory = scheduler_factory

        self.f1_source = torchmetrics.classification.MulticlassF1Score(num_classes=n_classes_source)
        self.f1_loc_source = torchmetrics.classification.BinaryF1Score()
        self.f1_source_per_class = torchmetrics.classification.MulticlassF1Score(
            num_classes=n_classes_source, average="none"
        )

        self.f1_target = torchmetrics.classification.MulticlassF1Score(num_classes=n_classes_target)
        self.f1_target_per_class = torchmetrics.classification.MulticlassF1Score(
            num_classes=n_classes_target, average="none"
        )
        self.f1_loc_target = torchmetrics.classification.BinaryF1Score()

        self.confusion_matrix_target = torchmetrics.classification.MulticlassConfusionMatrix(
            num_classes=n_classes_target, normalize="true"
        )

        self.target_conf_matrix_labels = target_conf_matrix_labels

        self.best_challenge_score_target = 0.0
        self.best_challenge_score_target_epoch = None

    @abstractmethod
    def forward_target(self, x: Tensor) -> Tensor:
        """
        Performs the forward pass of target domain input.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        ...

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the module.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        return self.forward_target(x)

    def training_step(self, batch: tuple[list[Tensor], list[Tensor]], batch_idx: int):
        """
        Perform a single training step.

        Args:
            batch: A tuple containing two lists of Tensors representing the source and target batches.
            batch_idx: An integer representing the index of the current batch.

        Returns:
            None
        """
        optimizer = self.optimizers()

        source_batch, target_batch = batch
        loss_source = self.inner.training_step(source_batch, batch_idx=batch_idx)
        self.manual_backward(loss_source)
        self.log("train_source_loss", loss_source, prog_bar=True, batch_size=batch[0][0].shape[0])

        t_img_pre, _, t_img_post, _ = target_batch
        target_preds = self.forward_target(torch.cat([t_img_pre, t_img_post], dim=1))
        target_preds_P = F.softmax(target_preds, dim=1)
        target_preds_labels = target_preds_P
        loss_target = self.msl_lambda * self.msl_loss(target_preds, target_preds_labels)
        self.manual_backward(loss_target)
        self.log("train_target_loss", loss_target, prog_bar=True, batch_size=batch[0][0].shape[0])

        optimizer.step()
        optimizer.zero_grad()

    def _do_eval_step(self, batch: tuple[list[Tensor], list[Tensor]]):
        """
        Perform a single evaluation step.

        Args:
            batch (tuple): A tuple containing two lists of Tensors representing the source and target batches.

        """
        source_batch, target_batch = batch
        s_img_pre, s_mask_pre, s_img_post, s_mask_post = source_batch
        t_img_pre, t_mask_pre, t_img_post, t_mask_post = target_batch

        source_preds = self.inner(torch.cat([s_img_pre, s_img_post], dim=1))
        source_preds_argmax = source_preds.argmax(dim=1)
        source_masks_post_argmax = s_mask_post.argmax(dim=1)
        self.f1_source(source_preds_argmax, source_masks_post_argmax)
        self.f1_source_per_class(source_preds_argmax, source_masks_post_argmax)
        self.f1_loc_source((source_preds_argmax > 0).to(torch.int), (source_masks_post_argmax > 0).to(torch.int))

        target_preds = self.forward_target(torch.cat([t_img_pre, t_img_post], dim=1))
        target_preds_argmax = target_preds.argmax(dim=1)
        target_masks_post_argmax = t_mask_post.argmax(dim=1)
        self.f1_target(target_preds_argmax, target_masks_post_argmax)
        self.f1_target_per_class(target_preds_argmax, target_masks_post_argmax)
        self.f1_loc_target((target_preds_argmax > 0).to(torch.int), (target_masks_post_argmax > 0).to(torch.int))

        self.confusion_matrix_target(target_preds_argmax, target_masks_post_argmax)

        common_log_kwargs = {
            "prog_bar": True,
            "batch_size": batch[0][0].shape[0],
            "on_epoch": True,
        }

        log_dict = {
            "f1_source": self.f1_source,
            "f1_loc_source": self.f1_loc_source,
            "f1_target": self.f1_target,
            "f1_loc_target": self.f1_loc_target,
        }

        self.log_dict(log_dict, **common_log_kwargs)

        for domain in ("source", "target"):
            for i in range(getattr(self, f"n_classes_{domain}")):
                metric_attr = f"f1_{domain}_per_class"
                self.log(
                    f"f1_{domain}_{i}", getattr(self, metric_attr)[i], metric_attribute=metric_attr, **common_log_kwargs
                )

    def _do_on_eval_epoch_end(self):
        """
        Perform evaluation operations at the end of each epoch.

        This method calculates the F1 scores for the source and target classes, as well as the F1 scores for the source and target locations.
        It then calculates the challenge scores for the source and target based on the F1 scores.
        The best challenge score for the target is updated if the current challenge score is higher.
        Finally, if the logger has the `log_image` attribute, it plots and logs the confusion matrix as an image.

        Note: This method does not specify argument and return types in the docstring.

        """
        f1_source_class = self.f1_source_per_class.compute()
        f1_loc_source = self.f1_loc_source.compute()
        f1_target_class = self.f1_target_per_class.compute()
        f1_loc_target = self.f1_loc_source.compute()

        f1_source_class = self.n_classes_source / sum([1 / f1_source_class[i] for i in range(self.n_classes_source)])
        challenge_score_source = 0.3 * f1_loc_source + 0.7 * f1_source_class
        self.log("f1_source_class", f1_source_class, prog_bar=True, on_epoch=True)
        self.log("challenge_score_source", challenge_score_source, prog_bar=True, on_epoch=True)

        f1_target_class = self.n_classes_target / sum([1 / f1_target_class[i] for i in range(self.n_classes_target)])
        challenge_score_target = 0.3 * f1_loc_target + 0.7 * f1_target_class
        self.log("f1_target_class", f1_target_class, prog_bar=True, on_epoch=True)
        self.log("challenge_score_target", challenge_score_target, prog_bar=True, on_epoch=True)

        if challenge_score_target >= self.best_challenge_score_target:
            self.best_challenge_score_target = challenge_score_target
            self.best_challenge_score_target_epoch = self.current_epoch
            self.log("best_challenge_score_target", challenge_score_target, prog_bar=True)
            self.log("best_challenge_score_target_epoch", self.current_epoch, prog_bar=True)

        if not hasattr(self.logger, "log_image"):
            return

        fig, ax = plt.subplots(figsize=(10, 10))

        self.confusion_matrix_target.plot(ax=ax, labels=self.target_conf_matrix_labels, cmap="Greens")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        im = torchvision.transforms.ToTensor()(Image.open(buf))
        self.logger.log_image(
            key="confusion_matrix",
            images=[im],
        )

    def on_train_epoch_end(self):
        """
        This method is called at the end of each training epoch.

        It performs the following steps:
        1. Retrieves the learning rate scheduler.
        2. Updates the learning rate using the scheduler.

        Note: This method does not take any arguments and does not return anything.
        """
        scheduler = self.lr_schedulers()
        scheduler.step()

    def test_step(self, batch: tuple[list[Tensor], list[Tensor]]):
        """
        Perform a test step on the given batch.

        Parameters:
        - batch: A tuple containing two lists of Tensors representing the input and target data.

        Returns:
        None
        """
        self._do_eval_step(batch)

    def on_test_epoch_end(self):
        """
        Perform actions at the end of each test epoch.

        This method is called at the end of each test epoch to perform any necessary actions or calculations.

        """
        self._do_on_eval_epoch_end()

    def validation_step(self, batch: list[Tensor], batch_idx: int):
        """
        Performs a validation step.

        Args:
            batch: The input batch of data.
            batch_idx: The index of the current batch.

        """
        self._do_eval_step(batch)

    def on_validation_epoch_end(self):
        """
        Perform any necessary operations at the end of each validation epoch.
        """
        self._do_on_eval_epoch_end()

    def configure_optimizers(self):
        """
        Configures the optimizers for the model.

        Returns:
            If a scheduler factory is provided, returns a tuple containing the optimizer and the scheduler.
            If no scheduler factory is provided, returns the optimizer.
        """
        optimizer = self.optimizer_factory(self.inner.model.parameters())
        if self.scheduler_factory:
            scheduler = self.scheduler_factory(optimizer)
            return [optimizer], [scheduler]
        else:
            return optimizer


class XBDMslModuleWrapper(BaseMSLModuleWrapper):
    def forward_target(self, x: Tensor) -> Tensor:
        return self.inner(x)


class FloodNetMslModuleWrapper(BaseMSLModuleWrapper):
    def forward_target(self, x: Tensor) -> Tensor:
        preds = self.inner(x)
        return torch.cat([preds[:, :2, ...], preds[:, 2:, ...].max(dim=1, keepdim=True).values], dim=1)


class RescueNetMslModuleWrapper(BaseMSLModuleWrapper):
    def forward_target(self, x: Tensor) -> Tensor:
        return self.inner(x)
