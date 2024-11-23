#!/usr/bin/env python3

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Collection, Literal

import numpy as np
import torch
import torchvision.transforms.functional as T
from PIL import Image
from torchvision.io import read_image  # type: ignore[import-untyped]
from torchvision.transforms import InterpolationMode
from torchvision.utils import draw_segmentation_masks
from tqdm import tqdm  # type: ignore[import-untyped]

DRY_RUN = False

def _resize_crop(img: torch.Tensor, out_size: int, interpolation: InterpolationMode) -> torch.Tensor:
    img_h = img.shape[1]
    img_w = img.shape[2]
    left = (img_w - img_h) // 2
    # assuming W > H
    return T.resized_crop(img, top=0, left=left, height=3000, width=3000, size=(out_size, out_size), interpolation=interpolation)

def load_img(img_path: Path | str, out_size: tuple[int, int]) -> torch.Tensor:
    img = read_image(str(img_path)).to(torch.float) / 255
    return _resize_crop(img, out_size, InterpolationMode.BILINEAR)

def load_mask(mask_path: Path | str, out_size: tuple[int, int]) -> torch.Tensor:
    _mask = read_image(str(mask_path))
    mask = _resize_crop(_mask, out_size, InterpolationMode.NEAREST)
    # Total class: 10 (
        # 'Background':0,
        # 'Building-flooded':1,
        # 'Building-non-flooded':2,
        # 'Road-flooded':3,
        # 'Road-non-flooded':4,
        # 'Water':5,
        # 'Tree':6,
        # 'Vehicle':7,
        # 'Pool':8,
        # 'Grass':9).
    # Output mask has 3 channels - 0. background (not building), 1. building non flooded, 2. building flooded
    mask_adj = torch.cat([(mask == 0).logical_or(mask >= 3), mask == 2, mask == 1])
    return mask_adj

def plot_mask(img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    colors = [
        (128, 128, 128),
        (0, 255, 0),
        (255, 0, 0),
    ]

    return draw_segmentation_masks((img * 255).to(torch.uint8), mask, alpha=0.3, colors=colors)

def img_path_to_labels_path(img_path: Path) -> Path:
    return Path(str(img_path.parents[0]).replace("-org-img", "-label-img"))  / f"{img_path.stem}_lab.png"


if __name__ == "__main__":
    DATASET_PATH = Path("data/floodnet/FloodNet-Supervised_v1.0")

    rglob = "*/*.jpg"
    num_images = len(list(DATASET_PATH.rglob(rglob)))

    todos_t = Literal["images", "masks", "preview"]

    def process_single_image(img_path: Path, out_size: int, todos: Collection[todos_t], out_path: Path) -> None:
        out_path_img = out_path / img_path.relative_to(img_path.parents[3])
        out_path_mask = img_path_to_labels_path(out_path_img)
        out_path_preview = out_path / "jpegs" / img_path.relative_to(img_path.parents[3])

        mask_path = img_path_to_labels_path(img_path)

        image = load_img(img_path, out_size)
        image_uint = (image * 255).to(torch.uint8)
        mask = load_mask(mask_path, out_size)

        preview = plot_mask(image, mask)

        _todos = (
            (
                [
                    (
                        np.asarray(image_uint.moveaxis(0, -1)),
                        out_path_img,
                        lambda path, arr: Image.fromarray(arr).save(
                            path.with_suffix(".png"), compress_level=0, optimize=False
                        ),
                    )
                ]
                if "images" in todos
                else []
            )
            + (
                [(np.asarray(mask.moveaxis(0, -1)), out_path_mask, lambda path, arr: np.savez_compressed(path.with_suffix(".npz"), arr))]
                if "masks" in todos
                else []
            )
            + (
                [
                    (
                        np.asarray(preview.moveaxis(0, -1)),
                        out_path_preview,
                        lambda path, arr: Image.fromarray(arr).save(path.with_suffix(".jpeg"), quality=70),
                    )
                ]
                if "preview" in todos
                else []
            )
        )

        global DRY_RUN
        for data, out_path_base, save in _todos:
            if not DRY_RUN:
                os.makedirs(out_path_base.parent, exist_ok=True)
                save(out_path_base, data)

        pass



    def wrapper(img_path: Path, out_size: int, todos: Collection[todos_t], out_path: Path) -> Path:
        process_single_image(img_path, out_size, todos, out_path)
        return img_path

    all_todos = {"images", "masks", "preview"}
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-size", type=int, help="Target image size")
    parser.add_argument("--out-path", type=Path, help="Output directory")
    parser.add_argument("--todos", nargs="+", choices=all_todos | {"all"})
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, help="Do not perform any writes")
    parser.add_argument("--workers", type=int, default=None, help="How many worker processes to use (defaults to number of cpus)")
    args = parser.parse_args()
    todos = set(args.todos) if "all" not in args.todos else all_todos
    DRY_RUN = args.dry_run
    if DRY_RUN:
        print("### DRY RUN! ###")

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        tasks = [executor.submit(wrapper, img_path, args.out_size, todos, args.out_path) for img_path in DATASET_PATH.rglob(rglob)]

        try:
            for task in tqdm(as_completed(tasks), total=num_images, desc=f"Creating {', '.join(todos)}"):
                tqdm.write(str(task.result()))
        except KeyboardInterrupt:
            for task in tasks:
                task.cancel()
