from typing import Callable, Optional

import torch
import torch.nn as nn
from torch import Tensor

from inz.models.base_pl_module import BasePLModule


class BaselineModule(BasePLModule):
    """PL wrapper for the Strong Baseline model."""

    def __init__(
        self,
        model: nn.Module,
        loss: nn.Module,
        optimizer_factory: Callable,
        scheduler_factory: Optional[Callable] = None,
        class_weights: Optional[Tensor] = None,
        n_classes: int = 5,
    ) -> None:
        """
        Initializes the BaselineModule.

        Args:
            model: The neural network model.
            loss: The loss function.
            optimizer_factory: A callable that creates an optimizer.
            scheduler_factory: A callable that creates a learning rate scheduler (optional).
            class_weights: The weights for each class (optional).
            n_classes: The number of classes (default is 5).
        """
        super(BaselineModule, self).__init__(
            model=model,
            optimizer_factory=optimizer_factory,
            scheduler_factory=scheduler_factory,
            class_weights=class_weights,
            n_classes=n_classes,
        )

        self.loss_fn = loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)  # type: ignore[no-any-return]

    def loss(
        self, images_pre: Tensor, masks_pre: Tensor, images_post: Tensor, masks_post: Tensor
    ) -> tuple[Tensor, Tensor]:
        preds = self.forward(torch.cat([images_pre, images_post], dim=1))
        if self.class_weights is not None:
            # per-class loss in unweighted!
            class_loss = torch.stack(
                [self.loss_fn(preds[:, i, ...], masks_post.to(torch.float)[:, i, ...]) for i in range(preds.shape[1])]
            )
            loss = class_loss.dot(self.class_weights).sum()
        else:
            loss = self.loss_fn(preds, masks_post.to(torch.float))
            class_loss = Tensor([0, 0, 0, 0, 0])

        return loss, class_loss
