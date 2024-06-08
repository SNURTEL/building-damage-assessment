#!/usr/bin/env python3

import argparse
import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from enum import Enum
from pathlib import Path
from typing import Collection, Literal

import geotiff  # type: ignore[import-untyped]
import numpy as np
import rasterio.features  # type: ignore[import-untyped]
import shapely.geometry  # type: ignore[import-untyped]
import shapely.wkt  # type: ignore[import-untyped]
from patchify import patchify  # type: ignore[import-untyped]
from PIL import Image, ImageDraw
from tqdm import tqdm  # type: ignore[import-untyped]


class BuildingDamage(str, Enum):
    no_damage = "no-damage"
    minor_damage = "minor-damage"
    major_damage = "major-damage"
    destroyed = "destroyed"
    un_classifeid = "un-classified"


_damage_to_label = {
    BuildingDamage.no_damage: 0,
    BuildingDamage.minor_damage: 1,
    BuildingDamage.major_damage: 2,
    BuildingDamage.destroyed: 3,
    BuildingDamage.un_classifeid: 4,
}


def load_geotiff(
    img_path: Path | str,
) -> Image.Image:
    img_path = Path(img_path)
    geotiff_img = geotiff.GeoTiff(img_path)
    array_img = np.array(geotiff_img.read(), dtype=np.uint8)
    return Image.fromarray(array_img)  # type: ignore[no-any-return]


def plot_ploygons(
    img: Image.Image,
    labels_path: Path | str,
    damage_to_rgba: dict[BuildingDamage, tuple[int, int, int, int]] | None = None,
) -> Image.Image:
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

    damage_to_rgba = damage_to_rgba or {
        BuildingDamage.no_damage: (0, 255, 0, 60),
        BuildingDamage.minor_damage: (244, 255, 0, 80),
        BuildingDamage.major_damage: (255, 174, 0, 80),
        BuildingDamage.destroyed: (255, 0, 0, 80),
        BuildingDamage.un_classifeid: (255, 255, 255, 80),
    }

    img = img.copy()

    with open(labels_path, mode="r") as fp:
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


def make_mask(labels_path: Path | str, img_size: tuple[int, int]) -> np.ndarray:
    """
    Create a binary-valued mask of dtype np.uint8 and shape (x, y, 5) given labels path.
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
    return np.stack(rasterized_layers).transpose((1, 2, 0))  # type: ignore[no-any-return]


def img_path_to_labels_path(img_path: Path) -> Path:
    return img_path.parents[1] / "labels" / img_path.with_suffix(".json").name


if __name__ == "__main__":
    DATASET_PATH = Path("data/xBD/geotiffs")

    OUT_PATH = Path("data/xBD_processed")
    OUT_MASKS_PATH = OUT_PATH / "masks"
    OUT_IMAGES_PATH = OUT_PATH / "images"
    OUT_JPEGS_PATH = OUT_PATH / "jpegs"

    IMG_SIZE = 1024
    PATCH_SIZE = 512

    rglob = "*/*.tif"
    num_images = len(list(DATASET_PATH.rglob(rglob)))

    todos_t = Literal["images", "masks", "preview"]

    def process_single_image(img_path: Path, todos: Collection[todos_t]) -> None:
        image = load_geotiff(img_path)

        mask_path = img_path_to_labels_path(img_path)
        mask = make_mask(mask_path, img_size=(IMG_SIZE, IMG_SIZE))

        preview = plot_ploygons(image, mask_path)

        _todos = (
            (
                [
                    (
                        np.asarray(image),
                        OUT_IMAGES_PATH,
                        lambda path, arr: Image.fromarray(arr).save(
                            path.with_suffix(".png"), compress_level=0, optimize=False
                        ),
                    )
                ]
                if "images" in todos
                else []
            )
            + (
                [(mask, OUT_MASKS_PATH, lambda path, arr: np.savez_compressed(path.with_suffix(".npz"), arr.transpose((2, 0, 1))))]
                if "masks" in todos
                else []
            )
            + (
                [
                    (
                        np.asarray(preview),
                        OUT_JPEGS_PATH,
                        lambda path, arr: Image.fromarray(arr).save(path.with_suffix(".jpeg"), quality=70),
                    )
                ]
                if "preview" in todos
                else []
            )
        )

        for data, out_dir, save in _todos:
            patches = patchify(data, (PATCH_SIZE, PATCH_SIZE, data.shape[2]), PATCH_SIZE)
            n_patches_h, n_patches_w, *_ = patches.shape

            out_path_base = out_dir / img_path.relative_to(img_path.parents[2])
            os.makedirs(out_path_base.parent, exist_ok=True)
            out_paths = (
                out_path_base.with_stem(out_path_base.stem + f"_patch{i}") for i in range(n_patches_h * n_patches_w)
            )

            for patch_num, out_path in enumerate(out_paths):
                save(out_path, patches[patch_num % n_patches_h, patch_num // n_patches_w, 0])


    def wrapper(img_path: Path, todos: Collection[todos_t]) -> Path:
        process_single_image(img_path, todos)
        return img_path

    all_todos = {"images", "masks", "preview"}
    parser = argparse.ArgumentParser()
    parser.add_argument("todos", nargs="+", choices=all_todos | {"all"})
    args = parser.parse_args()
    todos = set(args.todos) if "all" not in args.todos else all_todos

    with ProcessPoolExecutor() as executor:
        tasks = [executor.submit(wrapper, img_path, todos) for img_path in DATASET_PATH.rglob(rglob)]

        try:
            for task in tqdm(as_completed(tasks), total=num_images, desc=f"Creating {', '.join(todos)}"):
                tqdm.write(str(task.result()))
        except KeyboardInterrupt:
            for task in tasks:
                task.cancel()
