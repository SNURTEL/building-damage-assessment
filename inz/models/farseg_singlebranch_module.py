from __future__ import annotations

import sys
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from inz.models.base_pl_module import BasePLModule

sys.path.append("inz/external/farseg")

from module.farseg import FarSeg
from module.loss import cosine_annealing, linear_annealing, poly_annealing

del sys.path[sys.path.index("inz/external/farseg")]


class FarSegSingleBranchModule(BasePLModule):
    def __init__(
        self,
        model: nn.Module,
        optimizer_factory: Callable[[Any], torch.optim.Optimizer],
        scheduler_factory: Callable[[Any], torch.optim.lr_scheduler.LRScheduler] | None = None,
        n_classes: int = 5,
        class_weights: Tensor | None = None,
    ):
        super(FarSegSingleBranchModule, self).__init__(
            model=model,
            optimizer_factory=optimizer_factory,
            scheduler_factory=scheduler_factory,
            class_weights=class_weights,
            n_classes=n_classes,
        )

    def forward(self, x: Tensor) -> Tensor:
        # skip the first 3 channels (first image)
        return self.model(x[:, 3:, ...])  # type: ignore[no-any-return]

    def loss(
        self, images_pre: Tensor, masks_pre: Tensor, images_post: Tensor, masks_post: Tensor
    ) -> tuple[Tensor, Tensor]:
        preds = self.forward(torch.cat([images_pre, images_post], dim=1))

        loss = self.model.module.config.loss.cls_weight * self.model.module.cls_loss(
            preds, torch.argmax(masks_post, dim=1)
        )
        class_loss = torch.zeros((self.n_classes,))

        return loss, class_loss


class SingleBranchFarSeg(nn.Module):
    def __init__(self, farseg_config: dict, n_classes: int, class_weights: Tensor | None = None):
        super(SingleBranchFarSeg, self).__init__()
        self.farseg_config = farseg_config
        self.n_classes = n_classes
        self.module = FarSeg(config=farseg_config)
        self.class_weights = class_weights

        # replace the loss method, since the original implementation does not allow class weights in cross entropy
        def _softmax_focalloss(y_pred, y_true, ignore_index=255, gamma=2.0, normalize=False):
            """

            Args:
                y_pred: [N, #class, H, W]
                y_true: [N, H, W] from 0 to #class
                gamma: scalar

            Returns:

            """
            losses = F.cross_entropy(y_pred, y_true, weight=class_weights, ignore_index=ignore_index, reduction="none")
            with torch.no_grad():
                p = y_pred.softmax(dim=1)
                modulating_factor = (1 - p).pow(gamma)
                valid_mask = ~y_true.eq(ignore_index)
                masked_y_true = torch.where(valid_mask, y_true, torch.zeros_like(y_true))
                modulating_factor = torch.gather(
                    modulating_factor, dim=1, index=masked_y_true.unsqueeze(dim=1)
                ).squeeze_(dim=1)
                scale = 1.0
                if normalize:
                    scale = losses.sum() / (losses * modulating_factor).sum()
            losses = scale * (losses * modulating_factor).sum() / (valid_mask.sum() + p.size(0))

            return losses

        def _annealing_softmax_focalloss(
            y_pred, y_true, t, t_max, ignore_index=255, gamma=2.0, annealing_function=cosine_annealing
        ):
            losses = F.cross_entropy(
                y_pred,
                y_true,
                weight=class_weights,
                ignore_index=ignore_index,
                reduction="none",
            )
            with torch.no_grad():
                p = y_pred.softmax(dim=1)
                modulating_factor = (1 - p).pow(gamma)
                valid_mask = ~y_true.eq(ignore_index)
                masked_y_true = torch.where(valid_mask, y_true, torch.zeros_like(y_true))
                modulating_factor = torch.gather(
                    modulating_factor, dim=1, index=masked_y_true.unsqueeze(dim=1)
                ).squeeze_(dim=1)
                normalizer = losses.sum() / (losses * modulating_factor).sum()
                scales = modulating_factor * normalizer
            if t > t_max:
                scale = scales
            else:
                scale = annealing_function(1, scales, t, t_max)
            losses = (losses * scale).sum() / (valid_mask.sum() + p.size(0))
            return losses

        def _cls_loss(y_pred, y_true):
            if "softmax_focalloss" in self.module.config:
                return _softmax_focalloss(
                    y_pred,
                    y_true.long(),
                    ignore_index=self.module.config.loss.ignore_index,
                    gamma=self.module.config.softmax_focalloss.gamma,
                    normalize=self.module.config.softmax_focalloss.normalize,
                )
            elif "annealing_softmax_focalloss" in self.module.config:
                func_dict = dict(cosine=cosine_annealing, poly=poly_annealing, linear=linear_annealing)
                return _annealing_softmax_focalloss(
                    y_pred,
                    y_true.long(),
                    self.module.buffer_step.item(),
                    self.module.config.annealing_softmax_focalloss.max_step,
                    self.module.config.loss.ignore_index,
                    self.module.config.annealing_softmax_focalloss.gamma,
                    func_dict[self.module.config.annealing_softmax_focalloss.annealing_type],
                )
            return F.cross_entropy(
                y_pred, y_true.long(), weight=class_weights, ignore_index=self.module.config.loss.ignore_index
            )

        self.module.cls_loss = _cls_loss

    def forward(self, x) -> torch.Tensor:
        feat_list = self.module.en(x)
        fpn_feat_list = self.module.fpn(feat_list)
        if "scene_relation" in self.module.config:
            c5 = feat_list[-1]
            c6 = self.module.gap(c5)
            refined_fpn_feat_list = self.module.sr(c6, fpn_feat_list)
        else:
            refined_fpn_feat_list = fpn_feat_list

        final_feat = self.module.decoder(refined_fpn_feat_list)
        cls_pred = self.module.cls_pred_conv(final_feat)
        return self.module.upsample4x_op(cls_pred)
