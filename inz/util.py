import os
from typing import Iterable

import numpy as np
import torch
import torchvision.transforms.functional as T  # type: ignore[import-untyped]
import wandb
from matplotlib import pyplot as plt
from pytorch_lightning.loggers import WandbLogger
from torch import Tensor as Ts
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.utils import draw_segmentation_masks, make_grid  # type: ignore[import-untyped]
from tqdm import tqdm


def get_loc_cls_weights(
    dataloader: DataLoader, device: torch.device | None = None, drop_unclassified_class: bool = False,
) -> tuple[Ts, Ts]:
    """Iterate over a DataLoader and compute weights for localization and classification tasks.
    - Loc weights are computed as sum(cls==0)/n, sum(cls>0)/n. Shape = (2).
    - Cls weights are computed as sum(cls==c)/n. Shape = (C).

    Weight for the last class is "unclassified", therefore it's weight will albways
    be 0 - we don't want the model to try to predict it.


    Args:
        dataloader: DataLoader to iterate over
        device: Device to move the computed weights to. Defaults to current device.

    Returns:
        A tuple of loc and cls weight tensors.
    """
    device = device or torch.cuda.current_device  # type: ignore[assignment]

    n_classes = 5 if drop_unclassified_class else 6

    aaa_loc = []
    aaa_cls = []
    for batch in tqdm(dataloader, desc="Computing class weights"):
        _, pre_masks, _, post_masks = batch
        counts_post = torch.bincount(post_masks.argmax(dim=1).reshape(-1), minlength=n_classes)
        aaa_cls.append(counts_post)
        counts_pre = torch.bincount(pre_masks.argmax(dim=1).reshape(-1), minlength=n_classes)
        aaa_loc.append(torch.tensor([counts_pre[0], counts_pre[1:].sum()]))

    loc_counts = torch.stack(aaa_loc).sum(dim=0).to(torch.float)
    cls_counts = torch.stack(aaa_cls).sum(dim=0).to(torch.float)

    loc_weights = loc_counts.sum() / loc_counts
    loc_weights = (loc_weights / loc_weights.sum()).to(device)
    cls_weights = cls_counts.sum() / cls_counts
    cls_weights = (cls_weights / cls_weights.sum()).to(device)
    if not drop_unclassified_class:
        cls_weights *= torch.tensor(
            [1, 1, 1, 1, 1, 0]  # last class is "unclassified"
        ).to(device)
    return loc_weights, cls_weights


def show(imgs: Ts | list[Ts], titles: str | list[str] | None = None) -> None:
    """Display given images in a column.

    Args:
        imgs: Images to display
        titles: (optional) A title for each image.
    """
    imgs_list = [imgs] if not isinstance(imgs, list) else imgs
    titles_list: list[str] | list[None] = [titles] if not isinstance(titles, list) else titles  # type: ignore[assignment]
    assert len(imgs_list) == len(titles_list) or titles_list == [
        None
    ], f"Got {len(imgs_list)} images and {len(titles_list)} titles"
    fig, axes = plt.subplots(ncols=len(imgs_list), squeeze=False)
    for i, img in enumerate(imgs_list):
        img = img.detach()
        img = T.to_pil_image(img)
        axes[0, i].imshow(np.asarray(img))
        axes[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        if titles:
            axes[0, i].set_title(titles_list[i])


def show_masks_comparison(
    images_pre: Ts,
    images_post: Ts,
    masks_pre: Ts,
    masks_post: Ts,
    preds: Ts,
    colors: Iterable[tuple[int, int, int]] | None = None,
    opacity: float = 0.3,
) -> None:
    colors = colors or [
        (128, 128, 128),
        (0, 255, 0),
        (244, 255, 0),
        (255, 174, 0),
        (255, 0, 0),
        (255, 255, 255),
    ]
    show(
        [
            make_grid((images_pre + 1) / 2, nrow=1),
            make_grid(
                [
                    draw_segmentation_masks(((i + 1) * 127.5).to(torch.uint8), m, colors=colors, alpha=opacity)
                    for i, m in zip(images_pre, masks_pre.to(torch.bool))
                ],
                nrow=1,
            ),
            make_grid((images_post + 1) / 2, nrow=1),
            make_grid(
                [
                    draw_segmentation_masks(((i + 1) * 127.5).to(torch.uint8), m, colors=colors, alpha=opacity)
                    for i, m in zip(images_post, masks_post.to(torch.bool))
                ],
                nrow=1,
            ),
            make_grid(
                [
                    draw_segmentation_masks(((i + 1) * 127.5).to(torch.uint8), m, colors=colors, alpha=opacity)
                    # todo remove argmax and moveaxis
                    for i, m in zip(images_post, F.one_hot(preds.argmax(dim=1)).to(torch.bool).moveaxis(-1, 1))
                ],
                nrow=1,
            ),
        ],
        titles=[
            "Source images (pre)",
            "Ground truth (pre)",
            "Source images (post)",
            "Ground truth (post)",
            "Predicted masks",
        ],
    )


def get_wandb_logger(
    run_name: str | None = None,
    dir: str | None = None,
    api_key: str | None = None,
    project: str = "inz",
    watch_model: bool = False,
    watch_model_log_frequency: int = 100,
    watch_model_model: torch.nn.Module | None = None,
) -> WandbLogger:
    if not (WANDB_API_KEY := api_key):
        assert (WANDB_API_KEY := os.getenv("WANDB_API_KEY")), "No API key specified"
    wandb.login(key=WANDB_API_KEY, verify=True)
    # log_model=True respects lightning's save_top_k, setting to 'all' logs all intermediate checkpoints
    wandb_logger = WandbLogger(project=project, log_model=True, save_dir=dir, name=run_name)
    if watch_model:
        assert watch_model_model, "When watch_model=True, a model must be provided"
        assert watch_model_log_frequency > 0, "When watch_model=True, logging frequency must be positive"
        wandb_logger.watch(watch_model_model, log_freq=watch_model_log_frequency)
    return wandb_logger


def nested_dict_to_tuples(d: dict) -> tuple:
    out = []
    for k, v in d.items():
        if isinstance(v, dict):
            out.append((k, nested_dict_to_tuples(v)))
        elif isinstance(v, list):
            out.append((k, tuple(v)))
        else:
            out.append((k, v))
    return tuple(out)
