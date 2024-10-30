import torch

from inz.models.baseline_module import BaselineModule
from inz.xview2_strong_baseline.legacy.zoo.models import Res34_Unet_Double


class BaselineSingleBranchModule(Res34_Unet_Double):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # reduce the filter number by half
        self.res = torch.nn.Conv2d(48, 5, 1, stride=1, padding=0)

    # Since this is no longer a siamese network, we forward the tensor only once
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.forward_once(x)
        return self.res(output)


class SingleBranchBaselinePLModule(BaselineModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pretend you didn't see anything
        return super().forward(x[:, 3:, ...])
