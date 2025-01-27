import sys

import torch

from inz.models.baseline_module import BaselineModule

sys.path.append("inz/external/xview2_strong_baseline")

from legacy.zoo.models import Res34_Unet_Double  # type: ignore # noqa: E402

del sys.path[sys.path.index("inz/external/xview2_strong_baseline")]


class BaselineSingleBranchModule(Res34_Unet_Double):
    """Modified version of the Strong Baseline model with a single branch."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # reduce the filter number by half
        self.res = torch.nn.Conv2d(48, 5, 1, stride=1, padding=0)

    # Since this is no longer a siamese network, we forward the tensor only once
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.forward_once(x)
        return self.res(output)


class SingleBranchBaselinePLModule(BaselineModule):
    """Modified version of the BaselineModule with a single branch."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ignore the first 3 channels (first image)
        return super().forward(x[:, 3:, ...])
