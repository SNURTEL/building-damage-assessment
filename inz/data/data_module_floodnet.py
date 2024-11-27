import itertools
import os
from collections import Counter
from pathlib import Path
from typing import Any, Literal, Sequence, Callable

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms  # type: ignore[import-untyped]
from torchvision.io import read_image  # type: ignore[import-untyped]
import torchvision.transforms as T

import inz.data.event
from inz.data.event import Event, Hold, Subset, Test, Tier1, Tier3

import gc


# Keep this number low! More workers will marginally improve performance
# at a cost of huge ram (and swap!!!) usage.
DATALOADER_WORKER_COUNT = 1

Split = Literal["train", "val", "test"]


class FloodNetDataset(Dataset):
    """Dataset class for the FloodNet dataset."""

    def __init__(
        self,
        image_dir: Path,
        mask_dir: Path,
        transform: list[Callable[[torch.Tensor], torch.Tensor]] | None = None,
        xbd_compat_mode: bool = True
    ) -> None:
        super(FloodNetDataset, self).__init__()

        self.image_dir = image_dir
        self.mask_dir = mask_dir

        assert {f"{p.stem}_lab" for p in image_dir.iterdir()} == {p.stem for p in mask_dir.iterdir()}

        self.image_paths = list(image_dir.iterdir())
        self.mask_paths = [
            Path(str(img_path.parents[0]).replace("-org-img", "-label-img"))  / f"{img_path.stem}_lab.npz"
            for img_path in self.image_paths
        ]

        self.normalize = transforms.Normalize(0.5, 0.5)
        self.transform = transform

        self.xbd_compat_mode = xbd_compat_mode

    def __getitem__(self, index: int) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        img = read_image(str(self.image_paths[index])).to(torch.float) / 255
        mask_arr = np.load(self.mask_paths[index])["arr_0"]
        mask = torch.tensor(mask_arr, dtype=torch.float).moveaxis(-1, 0)

        stacked = torch.cat([img, mask], dim=0)
        if self.transform:
            transformed = self.transform(stacked)
        else:
            transformed = stacked

        if self.xbd_compat_mode:
            return (
                torch.zeros(img.shape).float(),
                torch.zeros(mask.shape).long(),
                self.normalize(transformed[: img.shape[0]]).float(),
                transformed[mask.shape[0]:].long()
            )
        else:
            return (
                self.normalize(transformed[: img.shape[0]]).float(),
                transformed[mask.shape[0]:].long()
            )

    def __len__(self) -> int:
        return len(self.image_paths)


class FloodNetModule(pl.LightningDataModule):
    """DataModule for the FloodNet dataset."""

    def __init__(
        self,
        path: Path | str,
        train_batch_size: int | None = None,
        val_batch_size: int | None = None,
        test_batch_size: int | None = None,
        transform: list[Callable[[torch.Tensor], torch.Tensor]] | None = None,
    ) -> None:
        super(FloodNetModule, self).__init__()

        self.transform = transform

        self._path = Path(path)

        self._train_dataset: Dataset
        self._val_dataset: Dataset
        self._test_dataset: Dataset

        self._train_batch_size = train_batch_size
        self._val_batch_size = val_batch_size
        self._test_batch_size = test_batch_size

    def prepare_data(self) -> None:
        super().prepare_data()

        for subset in ("train", "val", "test"):
            setattr(self, f"_{subset}_dataset", FloodNetDataset(
                image_dir=self._path / subset / f"{subset}-org-img",
                mask_dir=self._path / subset / f"{subset}-label-img",
            ))

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        if not self._train_batch_size:
            raise RuntimeError(f"Requested train dataloader, but train batch size is {self._train_batch_size}")
        return DataLoader(
            self._train_dataset,
            batch_size=self._train_batch_size,
            num_workers=DATALOADER_WORKER_COUNT,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> TRAIN_DATALOADERS:
        if not self._val_batch_size:
            raise RuntimeError(f"Requested val dataloader, but val batch size is {self._val_batch_size}")
        return DataLoader(
            self._val_dataset,
            batch_size=self._val_batch_size,
            num_workers=DATALOADER_WORKER_COUNT,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self) -> TRAIN_DATALOADERS:
        if not self._test_batch_size:
            raise RuntimeError(f"Requested test dataloader, but test batch size is {self._test_batch_size}")
        return DataLoader(
            self._test_dataset,
            batch_size=self._test_batch_size,
            num_workers=DATALOADER_WORKER_COUNT,
            pin_memory=True,
            persistent_workers=True,
        )
