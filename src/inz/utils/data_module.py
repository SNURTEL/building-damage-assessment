from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Literal, Sequence

import pytorch_lightning as pl
from torch.utils.data import Dataset, random_split


class Event(Enum):
    guatemala_volcano = "guatemala-volcano"
    hurricane_florence = "hurricane-florence"
    hurricane_harvey = "hurricane-harvey"
    hurricane_matthew = "hurricane-matthew"
    hurricane_michael = "hurricane-michael"
    joplin_tornado = "joplin-tornado"
    lower_puna_volcano = "lower-puna-volcano"
    mexico_earthquake = "mexico-earthquake"
    midwest_flooding = "midwest-flooding"
    moore_tornado = "moore-tornado"
    nepal_flooding = "nepal-flooding"
    palu_tsunami = "palu-tsunami"
    pinery_bushfire = "pinery-bushfire"
    portugal_wildfire = "portugal-wildfire"
    santa_rosa_wildfire = "santa-rosa-wildfire"
    socal_fire = "socal-fire"
    sunda_tsunami = "sunda-tsunami"
    tuscaloosa_tornado = "tuscaloosa-tornado"
    woolsey_fire = "woolsey-fire"


class _SubsetBase:
    """Base class for dataset subset (challenge split)"""

    events: set[Event]


class Test(_SubsetBase):
    """Events from "Test" challenge split"""

    events = {
        Event.guatemala_volcano,
        Event.hurricane_florence,
        Event.hurricane_harvey,
        Event.hurricane_matthew,
        Event.hurricane_michael,
        Event.mexico_earthquake,
        Event.midwest_flooding,
        Event.palu_tsunami,
        Event.santa_rosa_wildfire,
        Event.socal_fire,
    }


class Hold(_SubsetBase):
    """Events from "Hold" challenge split"""

    events = {
        Event.guatemala_volcano,
        Event.hurricane_florence,
        Event.hurricane_harvey,
        Event.hurricane_matthew,
        Event.hurricane_michael,
        Event.mexico_earthquake,
        Event.midwest_flooding,
        Event.palu_tsunami,
        Event.santa_rosa_wildfire,
        Event.socal_fire,
    }


class Tier1(_SubsetBase):
    """Events from "Tier 1" challenge split"""

    events = {
        Event.guatemala_volcano,
        Event.hurricane_florence,
        Event.hurricane_harvey,
        Event.hurricane_matthew,
        Event.hurricane_michael,
        Event.mexico_earthquake,
        Event.midwest_flooding,
        Event.palu_tsunami,
        Event.santa_rosa_wildfire,
        Event.socal_fire,
    }


class Tier3(_SubsetBase):
    """Events from "Tier 3" challenge split"""

    events = {
        Event.joplin_tornado,
        Event.lower_puna_volcano,
        Event.moore_tornado,
        Event.nepal_flooding,
        Event.pinery_bushfire,
        Event.portugal_wildfire,
        Event.sunda_tsunami,
        Event.tuscaloosa_tornado,
        Event.woolsey_fire,
    }


Subset = type[Test | Hold | Tier1 | Tier3]

Split = Literal["train", "val", "test"]


class LazySegmentationDataset(Dataset):
    """Basic dataset for loading images and maska for semantic segmentation from disk"""

    def __init__(self, image_paths: Sequence[Path], mask_paths: Sequence[Path]) -> None:
        """Init the class

        Args:
            image_paths: Paths to image files
            mask_paths: Paths to mask files
        """
        super().__init__()
        assert len(image_paths) == len(mask_paths), "Image and mask paths must be of the same length"
        self._image_paths = image_paths
        self._mask_paths = mask_paths

    def __getitem__(self, index: int) -> tuple[Path, Path]:
        return self._image_paths[index], self._mask_paths[index]

    def __len__(self) -> int:
        return len(self._image_paths)


class XBDDataModule(pl.LightningDataModule):
    """DataModule for the xBD dataset."""

    def __init__(
        self,
        path: Path | str,
        events: dict[Subset, Sequence[Event]] | None = None,
        val_faction: float | None = None,
        test_fraction: float | None = None,
        split_events: dict[Split, dict[Subset, Sequence[Event]]] | None = None,
    ) -> None:
        """
        Init the class. Supports building splitting int train / val / test by either explicit
        event specification, or random split with given dataset fractions.
        Assumes the dataset is structured as follows:
        - path
          - geotiffs
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

        Raises:
            AssertionError: Argument validation failed
            RuntimeError: Unable to determine split method
        """

        # TODO group pre and post disaster

        self._split_by_fraction = val_faction is not None and test_fraction is not None and split_events is None
        self._split_by_event = (
            events is None and val_faction is None and test_fraction is None and split_events is not None
        )
        assert (
            self._split_by_fraction != self._split_by_event
        ), 'Either provide "val_fraction" + "test_fraction" + "events" (optionally) or "events_split" without "events"'

        self._path = Path(path)
        self._val_fraction = val_faction
        self._test_fraction = test_fraction
        self._split_events = split_events

        _subsets: list[Subset] = [Test, Hold, Tier1, Tier3]
        self._events = events or {s: list(s.events) for s in _subsets}

        self._train_dataset: Dataset
        self._val_dataset: Dataset
        self._test_dataset: Dataset

        if self._split_by_fraction:
            # silence mypy
            assert val_faction is not None
            assert test_fraction is not None

            assert val_faction + test_fraction <= 1, "val_fraction + test_fraction cannot be greater that 1"
            assert val_faction >= 0 and test_fraction >= 0, "val_fraction and test_fraction cannot be negative"

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

    def prepare_data(self) -> None:
        """Prepare the dataset. This function should only be called by the pytorch-lightning framework.

        Raises:
            RuntimeError: Unable to determine split method
        """
        super().prepare_data()

        if self._split_by_event:
            assert self._split_events
            self._train_dataset = LazySegmentationDataset(
                *zip(*self._get_image_mask_paths(self._split_events["train"]))
            )
            self._val_dataset = LazySegmentationDataset(*zip(*self._get_image_mask_paths(self._split_events["val"])))
            self._test_dataset = LazySegmentationDataset(*zip(*self._get_image_mask_paths(self._split_events["test"])))
        elif self._split_by_fraction:
            # silence mypy
            assert self._val_fraction is not None
            assert self._test_fraction is not None

            all_events_dataset = LazySegmentationDataset(*zip(*self._get_image_mask_paths(self._events)))

            val_size = int(len(all_events_dataset) * self._val_fraction)
            test_size = int(len(all_events_dataset) * self._test_fraction)
            train_size = len(all_events_dataset) - val_size - test_size

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
        return [
            (path, self._path / "masks" / path.relative_to(self._path / "geotiffs").with_suffix(".npz"))
            for subset, subset_events in subset_events.items()
            for event in subset_events
            for path in (self._path / "geotiffs" / subset.__name__.lower() / "images").iterdir()
            if path.name.startswith(event.value)
        ]


if __name__ == "__main__":
    # foo = XDBDataModule(
    #     path=Path("data/xBD"),
    #     split_events={
    #         "train": {Tier1: list(Tier1.events), Tier3: list(Tier3.events)},
    #         "val": {
    #             Hold: list(Hold.events),
    #         },
    #         "test": {
    #             Test: list(Test.events),
    #         },
    #     },
    # )
    # foo.prepare_data()

    foo = XBDDataModule(
        path=Path("data/xBD"),
        events={
            Tier1: [
                Event.hurricane_harvey,
                Event.santa_rosa_wildfire,
                Event.palu_tsunami,
            ],
            Tier3: list(Tier3.events),
            Hold: list(Hold.events),
            Test: list(Test.events),
        },
        val_faction=0.1,
        test_fraction=0.1,
    )
    foo.prepare_data()
