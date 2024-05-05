#!/usr/bin/env python3

import itertools
import json
import os
from pathlib import Path

import numpy as np
import rasterio.features
import shapely.geometry
import shapely.wkt

_damage_to_label = {
    "no-damage": 0,
    "minor-damage": 1,
    "major-damage": 2,
    "destroyed": 3,
    "un-classified": 4,
}


def make_mask(labels_path: Path | str, img_size: tuple[int, int]) -> np.array:
    """
    Create a binary-valued mask of dtype np.uint8 and shape (5, x, y) given labels path.
    Assume labels are structured as follows:
    {
        "features": {
            "xy": [
                {
                    "properties: {
                        "subtype": <BuildingDamage>,
                        ...,
                    },
                    "wkt": POLYGON "((x1 Y1, X2 Y2, ..., XN Yn))"
                },
                ...
            ],
            ...
        },
        ...
    }
    """

    with open(labels_path, mode="r") as fp:
        labels_json = json.load(fp)
    polys_damages = [
        (shapely.wkt.loads(polygon["wkt"]), _damage_to_label[polygon["properties"].get("subtype", "no-damage")])
        for polygon in labels_json["features"]["xy"]
    ]
    key = lambda x: x[1]
    grouped = {k: list(grouper) for k, grouper in itertools.groupby(sorted(polys_damages, key=key), key=key)}
    rasterized = {
        k: np.clip(
            rasterio.features.rasterize([a for a, _ in group], out_shape=img_size, dtype=np.uint8), a_min=0, a_max=1
        )
        for k, group in grouped.items()
    }

    rasterized_layers = [rasterized.get(i, np.zeros(img_size, dtype=np.uint8)) for i in range(5)]
    return np.stack(rasterized_layers)


if __name__ == "__main__":
    DATASET_PATH = Path("data/xBD/geotiffs")
    MASKS_PATH = Path("data/xBD/masks")

    rglob = "*/*.json"
    num_images = len(list(DATASET_PATH.rglob(rglob)))
    for i, p in enumerate(DATASET_PATH.rglob(rglob)):
        msg = f"[{i+1} / {num_images}] {p.relative_to(p.parents[2])}"
        print(f"{msg: <88}...", end="")

        mask = make_mask(p, img_size=(1024, 1024))
        mask_path = (MASKS_PATH / p.relative_to(DATASET_PATH)).with_suffix("")
        os.makedirs(mask_path.parent, exist_ok=True)
        np.savez_compressed(mask_path, mask)

        print(f"\r{msg: <88}DONE")
