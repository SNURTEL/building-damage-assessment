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

# Keep this number low! More workers will marginally improve performance
# at a cost of huge ram (and swap!!!) usage.
DATALOADER_WORKER_COUNT = 2

Split = Literal["train", "val", "test"]


class XBDDataset(Dataset):
    """Dataset class for the xBD dataset."""

    def __init__(
        self,
        image_paths: Sequence[Path],
        mask_paths: Sequence[Path],
        drop_unclassified_channel: bool = False,
        transform: list[Callable[[torch.Tensor], torch.Tensor]] | None = None,
    ) -> None:
        """Init the class

        Args:
            image_paths: Paths to image files
            mask_paths: Paths to mask files
            drop_unclassified_channel: Whether to discard the 6th mask channel ("unclassified" class)
        """
        super(XBDDataset, self).__init__()

        self.drop_unclassified_channel = drop_unclassified_channel

        assert len(image_paths) == len(mask_paths), "Image and mask paths must be of the same length"
        key = lambda p: "post_disaster" in p.name
        images_grouped = itertools.groupby(sorted(image_paths, key=key), key=key)
        self._image_paths_pre, self._image_paths_post = [sorted(grouper) for _, grouper in images_grouped]
        assert (
            len(self._image_paths_pre) == len(self._image_paths_post)
        ), f"Got a different number of pre ({len(self._image_paths_pre)}) and post ({len(self._image_paths_post)}) images"
        masks_grouped = itertools.groupby(sorted(mask_paths, key=key), key=key)
        self._mask_paths_pre, self._mask_paths_post = [sorted(grouper) for _, grouper in masks_grouped]
        assert len(self._mask_paths_pre) == len(
            self._mask_paths_post
        ), f"Got a different number of pre ({len(self._mask_paths_pre)}) and post ({len(self._mask_paths_post)}) masks"

        assert len(self._image_paths_pre) + len(self._image_paths_post) == len(image_paths)
        assert len(self._mask_paths_pre) + len(self._mask_paths_post) == len(mask_paths)

        self.normalize = transforms.Normalize(0.5, 0.5)
        self.transform = transform

    def __getitem__(self, index: int) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        image_pre = read_image(str(self._image_paths_pre[index])).to(torch.float) / 255
        mask_pre_arr = np.load(self._mask_paths_pre[index])["arr_0"]
        mask_pre = torch.tensor(mask_pre_arr, dtype=torch.float)
        image_post = read_image(str(self._image_paths_post[index])).to(torch.float) / 255
        mask_post_arr = np.load(self._mask_paths_post[index])["arr_0"]
        mask_post = torch.tensor(mask_post_arr, dtype=torch.float)

        if self.drop_unclassified_channel:
            mask_pre = mask_pre[:-1, ...]
            mask_post = mask_post[:-1, ...]

        stacked = torch.cat([image_pre, mask_pre, image_post, mask_post], dim=0)
        if self.transform:
            transformed = self.transform(stacked)
        else:
            transformed = stacked

        return (
            self.normalize(transformed[: image_pre.shape[0]]),
            transformed[image_pre.shape[0] : image_pre.shape[0] + mask_pre.shape[0]],
            self.normalize(
                transformed[
                    image_pre.shape[0] + mask_pre.shape[0] : image_pre.shape[0]
                    + mask_pre.shape[0]
                    + image_post.shape[0]
                ]
            ),
            transformed[image_pre.shape[0] + mask_pre.shape[0] + image_post.shape[0] :],
        )

        # if self.drop_unclassified_channel:
        #     return (
        #         self.image_transform(image_pre),
        #         mask_pre[:-1, ...],
        #         self.image_transform(image_post),
        #         mask_post[:-1, ...],
        #     )  # type: ignore[return-value]
        # else:
        #     return self.image_transform(image_pre), mask_pre, self.image_transform(image_post), mask_post  # type: ignore[return-value]

    def __len__(self) -> int:
        return len(self._image_paths_pre)


class XBDDataModule(pl.LightningDataModule):
    """DataModule for the xBD dataset."""

    @classmethod
    def create(cls, *args: list[Any], **kwargs: dict[str, Any]):  # type: ignore[no-untyped-def]
        # Just to make hydra happy with using enums as dict keys
        if kwargs.get("events"):
            kwargs["events"] = {
                getattr(inz.data.event, subset_string_k): [Event[event] for event in events_v]
                for subset_string_k, events_v in kwargs["events"].items()
            }

        if kwargs.get("split_events"):
            kwargs["split_events"] = {
                split_k: {  # type: ignore
                    getattr(inz.data.event, subset_string_k): [Event[event] for event in events_v]
                    for subset_string_k, events_v in subsets_v.items()  # type: ignore
                }
                for split_k, subsets_v in kwargs["split_events"].items()
            }

        return cls(*args, **kwargs)  # type: ignore

    def __init__(
        self,
        path: Path | str,
        events: dict[Subset, Sequence[Event]] | None = None,
        val_fraction: float | None = None,
        test_fraction: float | None = None,
        split_events: dict[Split, dict[Subset, Sequence[Event]]] | None = None,
        train_batch_size: int | None = None,
        val_batch_size: int | None = None,
        test_batch_size: int | None = None,
        drop_unclassified_channel: bool = False,
        transform: list[Callable[[torch.Tensor], torch.Tensor]] | None = None,
    ) -> None:
        """
        Init the class. Supports building splitting int train / val / test by either explicit
        event specification, or random split with given dataset fractions.
        Assumes the dataset is structured as follows:
        - path
          - images
            - hold
              - images
                - guatemala-volcano_00000001_pre_disaster.tif
                - guatemala-volcano_00000001_post_disaster.tif
                - guatemala-volcano_00000002_pre_disaster.tif
                - guatemala-volcano_00000002_post_disaster.tif
                - ...
                - hurricane-florence_00000001_pre_disaster.tif
                - hurricane-florence_00000001_post_disaster.tif
                - ...
              - labels
                - ... (unimportant)
            - test
              - like above
            - tier1
              - like above
            - tier3
              - like above
          - masks
            - hold
              - labels
                - guatemala-volcano_00000001_pre_disaster.npz
                - guatemala-volcano_00000001_post_disaster.npz
                - ...
            - ...

        Args:
            path: Dataset path
            events: If using split by fraction, a mapping of subset to list of events to use.
                If None, all available events. Defaults to None.
            val_faction: If using split by fraction, fraction of data to put in the validation
                dataset. Defaults to None.
            test_fraction: If using split by fraction, fraction of data to put in the test dataset. Defaults to None.
            split_events: If using explicit split by event, a mapping of split name (train / val / test) to
                a mapping of subset to list of events to use. Defaults to None.
            train_batch_size: Batch size to use during training. Defaults to None (raises if DataLoader is requested).
            val_batch_size: Batch size to use during validation. Defaults to None (raises if DataLoader is requested).
            test_batch_size: Batch size to use during testing. Defaults to None (raises if DataLoader is requested).
            drop_unclassified_channel: Whether to discard the 6th mask channel ("unclassified" class)

        Raises:
            AssertionError: Argument validation failed
            RuntimeError: Unable to determine split method
        """
        super(XBDDataModule, self).__init__()

        self.drop_unclassified_channel = drop_unclassified_channel
        self.transform = transform

        # TODO group pre and post disaster

        self._split_by_fraction = val_fraction is not None and test_fraction is not None and split_events is None
        self._split_by_event = (
            events is None and val_fraction is None and test_fraction is None and split_events is not None
        )
        assert (
            self._split_by_fraction != self._split_by_event
        ), 'Either provide "val_fraction" + "test_fraction" + "events" (optionally) or "events_split" without "events"'

        self._path = Path(path)
        self._val_fraction = val_fraction
        self._test_fraction = test_fraction
        self._split_events = split_events

        _subsets: list[Subset] = [Test, Hold, Tier1, Tier3]
        self._events = events or {s: list(s.events) for s in _subsets}

        self._train_dataset: Dataset
        self._val_dataset: Dataset
        self._test_dataset: Dataset

        if self._split_by_fraction:
            # silence mypy
            assert val_fraction is not None
            assert test_fraction is not None

            assert val_fraction + test_fraction <= 1, "val_fraction + test_fraction cannot be greater that 1"
            assert val_fraction >= 0 and test_fraction >= 0, "val_fraction and test_fraction cannot be negative"

            for subset, subset_events in self._events.items():
                assert (
                    set(subset_events) <= subset.events
                ), f"{set(subset_events) - set(subset.events)} do not belong to {subset.__name__}"
        elif self._split_by_event:
            # silence mypy
            assert split_events

            flattened = [
                (subset, event)
                for split_subsets in split_events.values()
                for subset, subset_events in split_subsets.items()
                for event in subset_events
            ]
            duplicates = [item for item, count in Counter(flattened).items() if count > 1]
            assert not duplicates, f"{duplicates} are preset in more than one set in events_split"
        else:
            raise RuntimeError("Cannot split by fraction or event")

        self._train_batch_size = train_batch_size
        self._val_batch_size = val_batch_size
        self._test_batch_size = test_batch_size

    def prepare_data(self) -> None:
        """Prepare the dataset. This function should only be called by the pytorch-lightning framework.

        Raises:
            RuntimeError: Unable to determine split method
        """
        super().prepare_data()

        if self._split_by_event:
            assert self._split_events
            if self._split_events.get("train"):
                self._train_dataset = XBDDataset(  # type: ignore[arg-type, misc]
                    *zip(*self._get_image_mask_paths(self._split_events["train"])),  # type: ignore[arg-type]
                    drop_unclassified_channel=self.drop_unclassified_channel,
                    transform=self.transform,
                )
            if self._split_events.get("val"):
                self._val_dataset = XBDDataset(  # type: ignore[arg-type, misc]
                    *zip(*self._get_image_mask_paths(self._split_events["val"])),  # type: ignore[arg-type]
                    drop_unclassified_channel=self.drop_unclassified_channel,
                    transform=self.transform,
                )
            if self._split_events.get("test"):
                self._test_dataset = XBDDataset(  # type: ignore[arg-type, misc]
                    *zip(*self._get_image_mask_paths(self._split_events["test"])),  # type: ignore[arg-type]
                    drop_unclassified_channel=self.drop_unclassified_channel,
                    transform=self.transform,
                )
        elif self._split_by_fraction:
            # silence mypy
            assert self._val_fraction is not None
            assert self._test_fraction is not None

            all_events_dataset = XBDDataset(  # type: ignore[misc]
                *zip(*self._get_image_mask_paths(self._events)),  # type: ignore[arg-type, misc]
                drop_unclassified_channel=self.drop_unclassified_channel,
                transform=self.transform,
            )

            assert len(self._get_image_mask_paths(self._events)) == len(set(self._get_image_mask_paths(self._events)))

            val_size = int(len(all_events_dataset) * self._val_fraction)
            test_size = int(len(all_events_dataset) * self._test_fraction)
            train_size = len(all_events_dataset) - val_size - test_size

            assert train_size + val_size + test_size == len(all_events_dataset)

            self._train_dataset, self._val_dataset, self._test_dataset = random_split(
                all_events_dataset, [train_size, val_size, test_size]
            )
        else:
            raise RuntimeError("Cannot split by fraction or event")

    def _get_image_mask_paths(self, subset_events: dict[Subset, Sequence[Event]]) -> list[tuple[Path, Path]]:
        """Generate paths to image and mask files for given events. Assumes the dataset structure described in __init__.

        Args:
            subset_events: A mapping of dataset subset to a list of events.

        Returns:
            A list of image path + mask path pairs for every photo of given events in subsets.
        """
        out = [
            (path, self._path / "masks" / path.relative_to(self._path / "images").with_suffix(".npz"))
            for subset, subset_events in subset_events.items()
            for event in subset_events
            for path in (self._path / "images" / subset.__name__.lower() / "images").iterdir()
            if path.name.startswith(event.value)
        ]
        # assert len({i for i, _ in out}) == len([i for i, _ in out])
        # assert len({m for _, m in out}) == len([m for _, m in out])
        return out

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
