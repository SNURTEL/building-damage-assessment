#!/usr/bin/env python3

import argparse
import os
import random
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

DATASET: Literal["floodnet", "rescuenet"] | None = None
DRY_RUN = False

# Apparently, some masks in FloodNet dataset are rotated by 180 degrees not to make things too easy
FLOODNET_ROTATED_MASKS = [
    7301, 7315, 7339,7370, 7423, 7450, 7577, 7581, 7583, 7584, 6708, 6710, 6711, 6712, 6713, 6714, 6715, 7240, 7267,
    7314, 7340, 7366, 7410, 7422, 7438, 7439, 7575, 7579, 7580, 7582, 7601, 7602, 6694, 6709, 7338, 7369, 7407, 7437,
    7455, 7576, 7578
]

def _resize_crop(img: torch.Tensor, out_size: int, interpolation: InterpolationMode) -> torch.Tensor:
    img_h = img.shape[1]
    img_w = img.shape[2]
    left = (img_w - img_h) // 2
    # assuming W > H
    return T.resized_crop(img, top=0, left=left, height=img_h, width=img_h, size=(out_size, out_size), interpolation=interpolation)

def load_img(img_path: Path | str, out_size: tuple[int, int]) -> torch.Tensor:
    img = read_image(str(img_path)).to(torch.float) / 255
    return _resize_crop(img, out_size, InterpolationMode.BILINEAR)

def remap_mask_floodnet(mask: torch.Tensor):
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
    return torch.cat([(mask == 0).logical_or(mask >= 3), mask == 2, mask == 1])


def remap_mask_rescuenet(mask: torch.Tensor):
    # Total class: 10 (
        # 'Background':0,
        # 'Water':1,
        # 'Building_No_Damange':2,
        # 'Building_Minor_Damage':3,
        # 'Building_Major_Damage':4,
        # 'Building_Total_Destruction':5,
        # 'Vehicle':6,
        # 'Road-Clear':7,
        # 'Riad-Blocked':8,
        # 'Tree':9
        # 'Pool': 10).
    # Output mask has 5 channels
    #    - 0. background (not building)
    #    - 1. building not damaged
    #    - 2. building minor damage
    #    - 3. building major damage
    #    - 4. building destroyed
    return torch.cat([(mask <= 1).logical_or(mask >= 6), mask == 2, mask == 3, mask == 4, mask == 5])


def load_mask(mask_path: Path | str, out_size: tuple[int, int]) -> torch.Tensor:
    _mask = read_image(str(mask_path))
    mask = _resize_crop(_mask, out_size, InterpolationMode.NEAREST)
    return mask

def plot_mask(img: torch.Tensor, mask: torch.Tensor, colors: list[tuple[int, int, int]]) -> torch.Tensor:
    return draw_segmentation_masks((img * 255).to(torch.uint8), mask, alpha=0.3, colors=colors)

def img_path_to_labels_path(img_path: Path) -> Path:
    return Path(str(img_path.parents[0]).replace("-org-img", "-label-img"))  / f"{img_path.stem}_lab.png"


if __name__ == "__main__":
    if DATASET == "floodnet":
        DATASET_PATH = Path("data/floodnet/FloodNet-Supervised_v1.0")
    else:
        DATASET_PATH = Path("data/rescuenet/RescueNet")

    rglob = "*/*.jpg"
    num_images = len(list(DATASET_PATH.rglob(rglob)))

    todos_t = Literal["images", "masks", "preview"]

    if DATASET == "floodnet":
        remap_mask_fn = remap_mask_floodnet
        mask_colors = [
            (128, 128, 128),
            (0, 255, 0),
            (255, 0, 0),
        ]
    else:
        remap_mask_fn = remap_mask_rescuenet
        mask_colors = [
            (128, 128, 128),
            (0, 255, 0),
            (244, 255, 0),
            (255, 174, 0),
            (255, 0, 0),
        ]


    def process_single_image(img_path: Path, out_size: int, todos: Collection[todos_t], out_path: Path) -> None:
        out_path_img = out_path / img_path.relative_to(img_path.parents[3])
        out_path_mask = img_path_to_labels_path(out_path_img)
        out_path_preview = out_path / "jpegs" / img_path.relative_to(img_path.parents[3])

        mask_path = img_path_to_labels_path(img_path)

        _mask = load_mask(mask_path, out_size)
        mask = remap_mask_fn(_mask)

        if DATASET == "floodnet":
            if mask[1:3].sum() == 0 and random.random() > 0.1:
                print(f"SKIP {img_path.stem}")
                return

            if int(img_path.stem) in FLOODNET_ROTATED_MASKS:
                print(f"ROTATE {mask_path.stem}")
                # fix rotated masks
                mask = T.rotate(mask, angle=180)

        image = load_img(img_path, out_size)
        image_uint = (image * 255).to(torch.uint8)

        preview = plot_mask(image, mask, colors=mask_colors)

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
    parser.add_argument("--dataset", choices=("flodnet", "rescuenet"), nargs=1, help="Dataset to process", required=True)
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

    DATASET = args.dataset[0]
    print(f"DATASET: {DATASET}")

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        tasks = [executor.submit(wrapper, img_path, args.out_size, todos, args.out_path) for img_path in DATASET_PATH.rglob(rglob)]

        try:
            for task in tqdm(as_completed(tasks), total=num_images, desc=f"Creating {', '.join(todos)}"):
                tqdm.write(str(task.result()))
        except KeyboardInterrupt:
            for task in tasks:
                task.cancel()
