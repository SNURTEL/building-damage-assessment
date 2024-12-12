import argparse
import datetime
import importlib
import os
import sys
from functools import partial, partialmethod
from pathlib import Path
from types import SimpleNamespace
from typing import Type

import dotenv
import hydra
import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from hydra import compose, initialize

if Path.cwd().stem == "scripts":
    PROJECT_DIR = Path.cwd().parent
    os.chdir("..")
else:
    PROJECT_DIR = Path.cwd()

sys.path.append(str(PROJECT_DIR))

from inz.data.data_module import XBDDataModule
from inz.data.data_module_floodnet import FloodNetModule
from inz.data.event import Event, Hold, Tier1, Tier3, Test
from inz.models.base_pl_module import BasePLModule
from inz.util import get_wandb_logger

sys.path.append("inz/farseg")
sys.path.append("inz/dahitra")

dotenv.load_dotenv()
RANDOM_SEED = 123
pl.seed_everything(RANDOM_SEED)
device = torch.device("cuda")
torch.set_float32_matmul_precision("high")

# ##################################### CONFIG ##############################################

OPTIM_FACTORY = partial(torch.optim.AdamW, lr=0.00002, weight_decay=1e-6)
SCHED_FACTORY = partial(
    torch.optim.lr_scheduler.MultiStepLR,
    gamma=0.5,
    milestones=[5, 11, 17, 23, 29, 33, 47, 50, 60, 70, 90, 110, 130, 150, 170, 180, 190],
)
FLOODNET_CLASS_WEIGHTS = torch.Tensor([0.01, 0.5, 1.]).to(device)

#############################################################################################

def _floodnet_module_adapter(model_class: Type) -> Type:
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        preds = self.model(x)
        return torch.cat([preds[:, :2, ...], preds[:, 2:, ...].max(dim=1, keepdim=True).values], dim=1)

    class_init = model_class.__init__

    # This is not how this works. This is not how any of this works.
    base_class = model_class.__bases__[0]
    # base_class = [c for c in model_class.__mro__ if c is BasePLModule][0]

    model_class.__init__ = partialmethod(class_init, n_classes=3)
    base_class.forward = forward
    return model_class

def get_model(ckpt_path: Path, cfg: dict, wrap_floodnet: bool) -> BasePLModule:
    model_class_str = cfg["module"]["module"]["_target_"]
    model_class_name = model_class_str.split(".")[-1]
    module_path = ".".join(model_class_str.split(".")[:-1])
    imported_module = importlib.import_module(module_path)
    model_class = getattr(imported_module, model_class_name)

    if wrap_floodnet:
        model_class = _floodnet_module_adapter(model_class)

    model_partial = hydra.utils.instantiate(cfg["module"]["module"])

    model = model_class.load_from_checkpoint(ckpt_path, *model_partial.args, **model_partial.keywords).to(device)
    model.optimizer_factory = OPTIM_FACTORY
    model.scheduler_factory = SCHED_FACTORY

    if wrap_floodnet:
        model.class_weights = FLOODNET_CLASS_WEIGHTS
    
    return model

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

def get_floodnet_datamodule(batch_size: int, num_workers: int = 2) -> FloodNetModule:
    dm = FloodNetModule(
        path=Path(PROJECT_DIR / "data/floodnet_processed_512/FloodNet-Supervised_v1.0"),
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


def parse_args() -> SimpleNamespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--hydra-config", help="Hydra config dumped in training process", required=True)
    parser.add_argument("-c", "--checkpoint-path", help="Checkpoint to use", required=True)
    parser.add_argument(
        "-r", "--run-name", help="Run name; defaults to t_{original_run_name}", required=False, default=None
    )
    parser.add_argument("-e", "--events", help="Events to test on; comma-separated list of events", default=None)
    parser.add_argument(
        "-n",
        "--num-epochs",
        help="For how many epochs to finetune the model (default=20)",
        required=False,
        type=int,
        default=20,
    )
    parser.add_argument("--offline", action=argparse.BooleanOptionalAction, help="Do not log to wandb", default=False)
    parser.add_argument("--floodnet", action=argparse.BooleanOptionalAction, help="Use the floodnet dataset", default=False)
    parser.add_argument(
        "--skip-initial",
        action=argparse.BooleanOptionalAction,
        help="Skip validation pass on the model before finetuning",
        default=False,
    )

    args = parser.parse_args()

    assert bool(args.events) != bool(args.floodnet), "Provide either --events or --floodnet"

    return args

def main():
    args = parse_args()
    with initialize(version_base="1.3", config_path=args.hydra_config):
        cfg = compose(config_name="config", overrides=[])


    if args.floodnet:
        dm = get_floodnet_datamodule(
            batch_size=cfg["datamodule"]["datamodule"]["train_batch_size"]
        )
    else:
        events = {Event(event_name) for event_name in args.events.replace("_", "-").split(",")}
        dm = get_xbd_datamodule(
            events=events,
            batch_size=cfg["datamodule"]["datamodule"]["train_batch_size"]
        )
    model = get_model(
        ckpt_path=Path(args.checkpoint_path).resolve(),
        cfg=cfg,
        wrap_floodnet=args.floodnet
    )

    experiment_name = args.run_name or "t_finetune_" + ("floodnet_" if args.floodnet else "") + f"{cfg['experiment_name']}"
    if not args.offline:
        wandb_logger = get_wandb_logger(
            run_name=experiment_name,
            project=cfg["project_name"],
            watch_model_model=model,
            dir=f"outputs/{experiment_name}",
        )
        wandb_logger.experiment.config["hydra_cfg"] = cfg
        wandb_logger.experiment.config["optim_factory"] = repr(OPTIM_FACTORY)
        wandb_logger.experiment.config["sched_factory"] = repr(SCHED_FACTORY)
        wandb_logger.experiment.config["dataset"] = "floodnet" if args.floodnet else "xbd"
        if args.floodnet:
            wandb_logger.experiment.config["class_weights"] = FLOODNET_CLASS_WEIGHTS
        if not args.floodnet:
            wandb_logger.experiment.config["finetine_events"] = list(sorted(events))


    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=args.num_epochs,
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
                monitor="challenge_score_safe",
                mode="max",
                filename=experiment_name
                + "-{epoch:02d}-{step:03d}-{challenge_score_target:.4f}-best-challenge",
                save_last=False,
            ),
        ],
        log_every_n_steps=15,
        logger=wandb_logger if not args.offline else None
    )

    if not args.skip_initial:
        trainer.validate(model=model, datamodule=dm)

    trainer.fit(model=model, datamodule=dm)

if __name__ == "__main__":
    main()