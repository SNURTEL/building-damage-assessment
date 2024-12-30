import argparse
import datetime
import importlib
import os
import sys
from argparse import ArgumentParser
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Literal

import dotenv
import hydra
import pytorch_lightning as pl
import torch
import torchvision.transforms as T
import wandb
from hydra import compose, initialize

if Path.cwd().stem == "scripts":
    PROJECT_DIR = Path.cwd().parent
    os.chdir("..")
else:
    PROJECT_DIR = Path.cwd()

sys.path.append(str(PROJECT_DIR))

from inz.data.data_module import XBDDataModule
from inz.data.data_module_frnet import FRNetModule
from inz.data.event import Event, Hold, Tier1, Tier3, Test
from inz.data.zipped_data_module import ZippedDataModule
from inz.models.msl.msl_loss import IW_MaxSquareloss
from inz.models.msl.msl_module_wrapper import FRNetMslModuleWrapper, XBDMslModuleWrapper
from inz.util import get_wandb_logger

sys.path.append("inz/farseg")
sys.path.append("inz/dahitra")






# ############################# TODO USE PROPER NAMES







# ##################################### CONFIG ##############################################

OPTIM_FACTORY = partial(torch.optim.AdamW, lr=0.00005, weight_decay=1e-6)
SCHED_FACTORY = partial(
    torch.optim.lr_scheduler.MultiStepLR,
    gamma=0.5,
    milestones=[5, 11, 17, 23, 29, 33, 47, 50, 60, 70, 90, 110, 130, 150, 170, 180, 190],
)
LOSS_FACTORY = partial(IW_MaxSquareloss, ignore_index=-1)
MSL_LAMBDA = 0.5

#############################################################################################


DATASET: Literal["floodnet", "xbd", "rescuenet"] | None = None


def get_xbd_datamodule(events: list[Event], batch_size: int, num_workers: int = 2, val_fraction: float = 0.5) -> XBDDataModule:
    events_config = {
        Tier1: [],
        Tier3: [],
        Test: [],
        Hold: []
    }
    for event in events:
        for split in (Tier1, Tier3, Test, Hold):
            if event in split.events:
                events_config[split].append(event)
    dm = XBDDataModule(
        path=PROJECT_DIR / "data/xBD_processed_512",
        drop_unclassified_channel=True,
        events=events_config,
        train_batch_size=batch_size,
        val_batch_size=batch_size,
        test_batch_size=batch_size,
        val_fraction=val_fraction,
        test_fraction=0.,
        num_workers=num_workers,
        transform=T.Compose(
            transforms=[
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply(
                    p=0.6, transforms=[T.RandomAffine(degrees=(-10, 10), scale=(0.9, 1.1), translate=(0.1, 0.1))]
                ),
            ]
        ),
    )
    return dm


def get_floodnet_datamodule(batch_size: int, num_workers: int = 2) -> FRNetModule:
    path = Path(PROJECT_DIR / "data/floodnet_processed_512/FloodNet-Supervised_v1.0")
    return _get_floodnet_rescuenet_datamodule(path=path, batch_size=batch_size, num_workers=num_workers)


def get_rescuenet_datamodule(batch_size: int, num_workers: int = 2) -> FRNetModule:
    path = Path(PROJECT_DIR / "data/rescuenet_processed_512/RescueNet")
    return _get_floodnet_rescuenet_datamodule(path=path, batch_size=batch_size, num_workers=num_workers)


def _get_floodnet_rescuenet_datamodule(path: Path, batch_size: int, num_workers: int) -> FRNetModule:
    dm = FRNetModule(
            path=path,
            train_batch_size=batch_size,
            val_batch_size=batch_size,
            test_batch_size=batch_size,
            transform=T.Compose(
                transforms=[
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomApply(
                        p=0.6, transforms=[T.RandomAffine(degrees=(-10, 10), scale=(0.9, 1.1), translate=(0.1, 0.1))]
                    ),
                ]
            ),
            num_workers=num_workers
        )
    return dm


def main() -> pl.Trainer:
    dotenv.load_dotenv()
    RANDOM_SEED = 123
    pl.seed_everything(RANDOM_SEED)
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")

    parser = ArgumentParser()
    parser.add_argument("-d", "--hydra-config", help="Hydra config dumped in training process", required=True)
    parser.add_argument("-c", "--checkpoint-path", help="Checkpoint to use", required=True)
    parser.add_argument("-f", "--floodnet", help="Use FloodNet dataset. Specifying either -f, -r or -e is required.", action=argparse.BooleanOptionalAction)
    parser.add_argument("-r", "--rescuenet", help="Use RescueNet dataset. Specifying either -f, -r or -e is required.", action=argparse.BooleanOptionalAction)
    parser.add_argument("-e", "--events", help="Events to test on; comma-separated list of events, Specifying either -f or -e is required.", default=None)
    parser.add_argument("-s", "--size", choices=("min", "max"), nargs=1, default="max", help="How to match \
                        dataset sizes: min to crop the larger dataset, max to repeat the smaller one")
    parser.add_argument(
        "--run-name", help="Run name; defaults to t_floodnet_{original_run_name}", required=False, default=None
    )
    parser.add_argument(
        "-n",
        "--num-epochs",
        help="For how many epochs to adapt the model (default=5)",
        required=False,
        type=int,
        default=5,
    )
    parser.add_argument("--offline", action=argparse.BooleanOptionalAction, help="Do not log to wandb", default=False)
    parser.add_argument(
        "--skip-initial",
        action=argparse.BooleanOptionalAction,
        help="Skip testing the model before adaptation",
        default=False,
    )

    args = parser.parse_args()

    assert bool(args.events) + bool(args.floodnet) + bool(args.rescuenet) == 1, "Provide exactly one of (--events, --floodnet, --rescuenet)"

    global DATASET
    if args.floodnet:
        DATASET = "floodnet"
    elif args.rescuenet:
        DATASET = "rescuenet"
    else:
        DATASET = "xbd"

    with initialize(version_base="1.3", config_path=args.hydra_config):
        cfg = compose(config_name="config", overrides=[])

    # ############### SETUP MODEL #################

    model_class_str = cfg["module"]["module"]["_target_"]
    model_class_name = model_class_str.split(".")[-1]
    module_path = ".".join(model_class_str.split(".")[:-1])
    imported_module = importlib.import_module(module_path)
    model_class = getattr(imported_module, model_class_name)
    model_partial = hydra.utils.instantiate(cfg["module"]["module"])

    _model = model_class.load_from_checkpoint(args.checkpoint_path, *model_partial.args, **model_partial.keywords).to(
        device
    )
    _model.class_weights = _model.class_weights.to(device)

    if DATASET == "floodnet":
        model = FRNetMslModuleWrapper(
            pl_module=_model,
            n_classes_target=3,
            msl_loss_module=LOSS_FACTORY(num_class=3).to(device),
            msl_lambda=MSL_LAMBDA,
            optimizer_factory=OPTIM_FACTORY,
            scheduler_factory=SCHED_FACTORY,
            target_conf_matrix_labels=("Background", "Non-flooded", "Flooded"),
        ).to(device)
    else:
        model = XBDMslModuleWrapper(
            pl_module=_model,
            n_classes_target=5,
            msl_loss_module=LOSS_FACTORY(num_class=5).to(device),
            msl_lambda=MSL_LAMBDA,
            optimizer_factory=OPTIM_FACTORY,
            scheduler_factory=SCHED_FACTORY,
            target_conf_matrix_labels=["Background", "No damage", "Minor damage", "Major damage", "Destroyed"],
        ).to(device)

    # ############### SETUP DATAMODULE #################

    BATCH_SIZE = int(cfg["datamodule"]["datamodule"]["train_batch_size"] * 0.9)

    _dm_source = hydra.utils.instantiate(cfg["datamodule"]["datamodule"])

    if DATASET == "floodnet":
        _dm_target = get_floodnet_datamodule(batch_size=BATCH_SIZE, num_workers=1)
    elif DATASET == "rescuenet":
        _dm_target = get_rescuenet_datamodule(batch_size=BATCH_SIZE, num_workers=1)
    else:
        events = {Event(event_name) for event_name in args.events.replace("_", "-").split(",")}
        _dm_target = get_xbd_datamodule(
            events=events,
            batch_size=cfg["datamodule"]["datamodule"]["train_batch_size"],
            val_fraction=0.
        )

    dm = ZippedDataModule(
        dm1=_dm_source,
        dm2=_dm_target,
        match_type=args.size[0],
        num_workers=3 if args.floodnet else 2,
        train_batch_size=BATCH_SIZE,
        val_batch_size=BATCH_SIZE,
        test_batch_size=BATCH_SIZE,
    )
    dm.prepare_data()
    dm.setup("fit")

    # ############### TEST AND ADAPT #################

    if not args.skip_initial:
        test_without_adaptation(model=model, datamodule=dm, cfg=cfg, offline=args.offline, run_name=args.run_name)

    train_adapt(model=model, datamodule=dm, n_epochs=args.num_epochs, cfg=cfg, offline=args.offline, run_name=args.run_name)

    return


def test_without_adaptation(model: FRNetMslModuleWrapper | XBDMslModuleWrapper, datamodule: ZippedDataModule, cfg: dict, offline: bool, run_name: str | None = None):
    dm = deepcopy(datamodule)
    dm.prepare_data()
    dm.setup("test")
    dm.test_dataloader = dm.val_dataloader

    if not offline:
        run_name = run_name or f"t_initial_{DATASET}_{cfg['experiment_name']}"
        wandb_logger = get_wandb_logger(
            run_name=run_name,
            project=cfg["project_name"],
            watch_model_model=model,
            dir=f"outputs/{run_name}",
        )
        wandb_logger.experiment.config["hydra_cfg"] = cfg

    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=1,
        precision="bf16-mixed",
        deterministic=True,
        sync_batchnorm=True,
        callbacks=[pl.callbacks.RichProgressBar()],
        logger=wandb_logger if not offline else None,
        log_every_n_steps=10
    )

    trainer.test(model=model, datamodule=dm)

    if not offline:
        wandb.finish()

def train_adapt(model: FRNetMslModuleWrapper, datamodule: ZippedDataModule, cfg: dict, offline: bool, n_epochs: int, run_name: str | None = None):
    experiment_name = run_name or f"t_{DATASET}_msl_{cfg['experiment_name']}"

    if not offline:
        wandb_logger = get_wandb_logger(
            run_name=experiment_name,
            project=cfg["project_name"],
            watch_model_model=model,
            dir="outputs/.wandb_tests",
        )
        wandb_logger.experiment.config["hydra_cfg"] = cfg
        wandb_logger.experiment.config["optim_factory"] = repr(OPTIM_FACTORY)
        wandb_logger.experiment.config["sched_factory"] = repr(SCHED_FACTORY)
        wandb_logger.experiment.config["loss_factory"] = repr(LOSS_FACTORY)

    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=n_epochs,
        precision="bf16-mixed",
        deterministic=True,
        sync_batchnorm=True,
        callbacks=[
            pl.callbacks.RichProgressBar(),
            pl.callbacks.ModelCheckpoint(
                dirpath=PROJECT_DIR
                / "outputs"
                / experiment_name
                / f"{datetime.datetime.now().replace(microsecond=0).isoformat()}"
                / "checkpoints",
                save_top_k=1,
                monitor="challenge_score_target",
                mode="max",
                filename=experiment_name
                + "-{epoch:02d}-{step:03d}-{challenge_score_target:.4f}-best-challenge-" + f"{DATASET}",
                save_last=False,
            ),
        ],
        logger=wandb_logger if not offline else None,
        log_every_n_steps=10
    )

    trainer.fit(model=model, datamodule=datamodule)

    if not offline:
        wandb.finish()


if __name__ == "__main__":
    main()
