#!/usr/bin/env python3

import json
import os
from enum import Enum
from pathlib import Path

import geotiff
import numpy as np
import shapely.geometry
import shapely.wkt
from PIL import Image, ImageDraw


class BuildingDamage(str, Enum):
    no_damage = "no-damage"
    minor_damage = "minor-damage"
    major_damage = "majod-damage"
    destroyed = "destroyed"
    un_classifeid = "un_classified"


def plot_ploygons(
    img_path: Path | str, damage_to_rgba: dict[BuildingDamage, tuple[int, int, int, int]] | None = None
) -> Image:
    """
    Plot damage polygons over GeoTIFF image from given file.
    Assume labels for <TIER>/images/<IMG>.tif are
    stored in <TIER>/labels/<IMG>.json and are structured as follows:
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
    img_path = Path(img_path)

    damage_to_rgba = damage_to_rgba or {
        "no-damage": (0, 255, 0, 60),
        "minor-damage": (244, 255, 0, 80),
        "major-damage": (255, 174, 0, 80),
        "destroyed": (255, 0, 0, 80),
        "un-classified": (255, 255, 255, 80),
    }

    geotiff_img = geotiff.GeoTiff(img_path)
    array_img = np.array(geotiff_img.read(), dtype=np.uint8)
    img = Image.fromarray(array_img)
    with open(img_path.parents[1] / "labels" / img_path.with_suffix(".json").name, mode="r") as fp:
        labels_json = json.load(
            fp,
        )
    polys_damages = [
        (shapely.wkt.loads(polygon["wkt"]), polygon["properties"].get("subtype", "no-damage"))
        for polygon in labels_json["features"]["xy"]
    ]
    draw = ImageDraw.Draw(img, mode="RGBA")

    for poly, damage in polys_damages:
        x, y = poly.exterior.coords.xy
        coords = list(zip(x, y))
        draw.polygon(coords, damage_to_rgba[damage])

    return img


if __name__ == "__main__":
    DATASET_PATH = Path("data/xBD/geotiffs")
    JPEGS_PATH = Path("data/xBD/jpgs")

    rglob = "*/*.tif"
    num_images = len(list(DATASET_PATH.rglob(rglob)))
    for i, p in enumerate(DATASET_PATH.rglob(rglob)):

        msg = f"[{i+1} / {num_images}] {p.relative_to(p.parents[2])}"
        print(f"{msg: <88}...", end='')

        img = plot_ploygons(p)
        img_path = (JPEGS_PATH / p.relative_to(DATASET_PATH)).with_suffix('.jpeg')
        os.makedirs(img_path.parent, exist_ok=True)
        img.save(img_path, quality=70)

        print(f"\r{msg: <88}DONE")

