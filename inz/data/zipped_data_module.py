from typing import Literal, Protocol

import pytorch_lightning as pl
import torch
import torch.utils
from torch.utils.data import DataLoader, Dataset


class BDADataModule(Protocol):
    _train_dataset: torch.utils.data.Dataset
    _val_dataset: torch.utils.data.Dataset
    _test_dataset: torch.utils.data.Dataset

    def prepare_data(self) -> None: ...

    def setup(self, stage: str | None = None) -> None: ...


class ZippedDataSet(torch.utils.data.Dataset):
    def __init__(self,
                 dataset1: Dataset,
                 dataset2: Dataset,
        match_type: Literal["min", "max"]) -> None:
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.match_type = match_type
        self.smaller_len = min([len(self.dataset1), len(self.dataset2)])
        self.larger_len = max([len(self.dataset1), len(self.dataset2)])
        self.len = self.smaller_len if match_type == "min" else self.larger_len

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if idx >= self.len:
            raise IndexError("Nope")

        idx = idx % self.smaller_len
        return (self.dataset1[idx], self.dataset2[idx])


class ZippedDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dm1: BDADataModule,
        dm2: BDADataModule,
        match_type: Literal["min", "max"],
        num_workers: int,
        train_batch_size: int | None = None,
        val_batch_size: int | None = None,
        test_batch_size: int | None = None,
    ) -> None:
        super(ZippedDataModule, self).__init__()
        self.dm1 = dm1
        self.dm2 = dm2
        self.match_type = match_type

        self._train_batch_size = train_batch_size
        self._val_batch_size = val_batch_size
        self._test_batch_size = test_batch_size

        self._train_dataset: torch.utils.data.Dataset
        self._val_dataset: torch.utils.data.Dataset
        self._test_dataset: torch.utils.data.Dataset

        self.num_workers = num_workers

    def prepare_data(self) -> None:
        self.dm1.prepare_data()
        self.dm2.prepare_data()

        # todo will not work with TensorDataset
        self._train_dataset = ZippedDataSet(self.dm1._train_dataset, self.dm2._train_dataset, match_type=self.match_type)
        # self._val_dataset = ZippedDataSet(self.dm1._val_dataset, self.dm2._val_dataset, match_type=self.match_type)
        # self._test_dataset = ZippedDataSet(self.dm1._test_dataset, self.dm2._test_dataset, match_type=self.match_type)

        # danger! In UDA, train = val = test
        self._val_dataset = ZippedDataSet(self.dm1._val_dataset, self.dm2._train_dataset, match_type=self.match_type)
        self._test_dataset = ZippedDataSet(self.dm1._test_dataset, self.dm2._train_dataset, match_type=self.match_type)

    def setup(self, stage: str | None = None) -> None:
        self.dm1.setup(stage)
        self.dm2.setup(stage)

    def train_dataloader(self):
        if not self._train_batch_size:
            raise RuntimeError(f"Requested train dataloader, but train batch size is {self._train_batch_size}")
        return DataLoader(
            self._train_dataset,
            batch_size=self._train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=False,
            persistent_workers=False,
        )

    def val_dataloader(self):
        if not self._val_batch_size:
            raise RuntimeError(f"Requested val dataloader, but val batch size is {self._val_batch_size}")
        return DataLoader(
            self._val_dataset,
            batch_size=self._val_batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=False,
        )

    def test_dataloader(self):
        if not self._test_batch_size:
            raise RuntimeError(f"Requested test dataloader, but test batch size is {self._test_batch_size}")
        return DataLoader(
            self._test_dataset,
            batch_size=self._test_batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=False,
        )
