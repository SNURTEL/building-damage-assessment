import os
import sys
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
from torch import Tensor

from inz.models.base_pl_module import BasePLModule

cwd = Path().resolve()
sys.path.append("inz/external/dahitra")

os.chdir(cwd / "inz/external/dahitra/xBD_code")

from zoo.model_transformer_encoding import BASE_Transformer_UNet  # type: ignore # noqa: E402, I001
from losses import ComboLoss  # type: ignore # noqa: E402, F401

os.chdir(cwd)

del sys.path[sys.path.index("inz/external/dahitra")]


class DahitraModule(nn.Module):
    def __init__(self, *args, image_size: int = 1024, **kwargs) -> None:
        super(DahitraModule, self).__init__()
        self.module = BASE_Transformer_UNet(*args, **kwargs)

        scale_factor = image_size / 1024

        # hack positional embedding sizes to enable images of other sizes than 1024px
        dim_5, dim_4, dim_3, dim_2 = 32, 32, 32, 32  # noqa: F841
        self.module.pos_embedding_decoder_5 = nn.Parameter(
            torch.randn(1, dim_5, int(16 * scale_factor), int(16 * scale_factor))
        )
        self.module.pos_embedding_decoder_4 = nn.Parameter(
            torch.randn(1, dim_4, int(32 * scale_factor), int(32 * scale_factor))
        )
        self.module.pos_embedding_decoder_3 = nn.Parameter(
            torch.randn(1, dim_3, int(64 * scale_factor), int(64 * scale_factor))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)


class DahitraPLModule(BasePLModule):
    def __init__(
        self,
        model: nn.Module,
        loss: nn.Module,
        optimizer_factory: Callable[[Any], torch.optim.Optimizer],
        scheduler_factory: Callable[[Any], torch.optim.lr_scheduler.LRScheduler] | None = None,
        class_weights: Tensor | None = None,
    ):
        super(DahitraPLModule, self).__init__(
            model=model,
            optimizer_factory=optimizer_factory,
            scheduler_factory=scheduler_factory,
            class_weights=class_weights,
        )

        self._loss = loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward(x)

    def loss(
        self, images_pre: Tensor, masks_pre: Tensor, images_post: Tensor, masks_post: Tensor
    ) -> tuple[Tensor, Tensor]:
        preds = self.forward(torch.cat([images_pre, images_post], dim=1))
        loss0 = self._loss(preds[:, 0, ...], masks_post[:, 0, ...])
        loss1 = self._loss(preds[:, 1, ...], masks_post[:, 1, ...])
        loss2 = self._loss(preds[:, 2, ...], masks_post[:, 2, ...])
        loss3 = self._loss(preds[:, 3, ...], masks_post[:, 3, ...])
        loss4 = self._loss(preds[:, 4, ...], masks_post[:, 4, ...])
        stacked = torch.stack([loss0, loss1, loss2, loss3, loss4])
        loss = (stacked * self.class_weights).sum()
        return loss, stacked
