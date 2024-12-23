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

        self.best_challenge_score_target = 0.
        self.best_challenge_score_target_epoch = None

    @abstractmethod
    def forward_target(self, x: Tensor) -> Tensor: ...

    def training_step(self, batch: tuple[list[Tensor], list[Tensor]], batch_idx: int):
        optimizer = self.optimizers()

        source_batch, target_batch = batch
        loss_source = self.inner.training_step(source_batch, batch_idx=batch_idx)
        self.manual_backward(loss_source)
        self.log("train_source_loss", loss_source, prog_bar=True, batch_size=batch[0][0].shape[0])

        t_img_pre, _, t_img_post, _ = target_batch
        target_preds = self.forward_target(torch.cat([t_img_pre, t_img_post], dim=1))
        target_preds_P = F.softmax(target_preds, dim=1)
        target_preds_labels = target_preds_P
        loss_target = self.msl_loss(target_preds, target_preds_labels)
        self.manual_backward(loss_target)
        self.log("train_target_loss", loss_target, prog_bar=True, batch_size=batch[0][0].shape[0])

        optimizer.step()
        optimizer.zero_grad()

    def _do_eval_step(self, batch: tuple[list[Tensor], list[Tensor]]):
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
                    f"f1_{domain}_{i}",
                    getattr(self, metric_attr)[i],
                    metric_attribute=metric_attr,
                    **common_log_kwargs
                )

    def _do_on_eval_epoch_end(self):
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

        if challenge_score_target > self.best_challenge_score_target:
            self.best_challenge_score_target = challenge_score_target
            self.best_challenge_score_target_epoch = self.current_epoch
            self.log("best_challenge_score_target", challenge_score_target, prog_bar=True)
            self.log("best_challenge_score_target_epoch", self.current_epoch, prog_bar=True)

        if not hasattr(self.logger, "log_image"):
            return

        fig, ax = plt.subplots(figsize=(10, 10))

        self.confusion_matrix_target.plot(
            ax=ax, labels=self.target_conf_matrix_labels, cmap="Greens"
        )

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        im = torchvision.transforms.ToTensor()(Image.open(buf))
        self.logger.log_image(
            key="confusion_matrix",
            images=[im],
        )

    def on_train_epoch_end(self):
        scheduler = self.lr_schedulers()
        scheduler.step()

    def test_step(self, batch: tuple[list[Tensor], list[Tensor]]):
        self._do_eval_step(batch)

    def on_test_epoch_end(self):
        self._do_on_eval_epoch_end()

    def validation_step(self, batch: list[Tensor], batch_idx: int):
        self._do_eval_step(batch)

    def on_validation_epoch_end(self):
        self._do_on_eval_epoch_end()

    def configure_optimizers(self):
        optimizer = self.optimizer_factory(self.inner.model.parameters())
        if self.scheduler_factory:
            scheduler = self.scheduler_factory(optimizer)
            return [optimizer], [scheduler]
        else:
            return optimizer


class XBDMslModuleWrapper(BaseMSLModuleWrapper):
    def forward_target(self, x: Tensor) -> Tensor:
        return self.inner(x)


class FloodnetMslModuleWrapper(BaseMSLModuleWrapper):
    def forward_target(self, x: Tensor) -> Tensor:
        preds = self.inner(x)
        return torch.cat([preds[:, :2, ...], preds[:, 2:, ...].max(dim=1, keepdim=True).values], dim=1)
