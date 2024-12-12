import argparse
import datetime
import importlib
import os
import sys
from argparse import ArgumentParser
from copy import deepcopy
from functools import partial
from pathlib import Path

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

from inz.data.data_module_floodnet import FloodNetModule
from inz.data.zipped_data_module import ZippedDataModule
from inz.models.msl.msl_loss import IW_MaxSquareloss
from inz.models.msl.msl_module_wrapper import FloodnetMslModuleWrapper
from inz.util import get_wandb_logger

sys.path.append("inz/farseg")
sys.path.append("inz/dahitra")


# ##################################### CONFIG ##############################################

OPTIM_FACTORY = partial(torch.optim.AdamW, lr=0.00005, weight_decay=1e-6)
SCHED_FACTORY = partial(
    torch.optim.lr_scheduler.MultiStepLR,
    gamma=0.5,
    milestones=[5, 11, 17, 23, 29, 33, 47, 50, 60, 70, 90, 110, 130, 150, 170, 180, 190],
)
LOSS_FACTORY = partial(IW_MaxSquareloss, ignore_index=-1, num_class=3, ratio=0.2)

#############################################################################################


def main() -> pl.Trainer:
    dotenv.load_dotenv()
    RANDOM_SEED = 123
    pl.seed_everything(RANDOM_SEED)
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")

    parser = ArgumentParser()
    parser.add_argument("-d", "--hydra-config", help="Hydra config dumped in training process", required=True)
    parser.add_argument("-c", "--checkpoint-path", help="Checkpoint to use", required=True)
    parser.add_argument(
        "-r", "--run-name", help="Run name; defaults to t_floodnet_{original_run_name}", required=False, default=None
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

    model = FloodnetMslModuleWrapper(
        pl_module=_model,
        n_classes_target=3,
        msl_loss_module=LOSS_FACTORY().to(device),
        msl_lambda=0.2,
        optimizer_factory=OPTIM_FACTORY,
        scheduler_factory=SCHED_FACTORY,
        target_conf_matrix_labels=("Background", "Non-flooded", "Flooded"),
    ).to(device)

    # ############### SETUP DATAMODULE #################

    BATCH_SIZE = cfg["datamodule"]["datamodule"]["train_batch_size"] // 2

    _dm_source = hydra.utils.instantiate(cfg["datamodule"]["datamodule"])
    _dm_target = FloodNetModule(
        path=Path(PROJECT_DIR / "data/floodnet_processed_512/FloodNet-Supervised_v1.0"),
        train_batch_size=BATCH_SIZE,
        val_batch_size=BATCH_SIZE,
        test_batch_size=BATCH_SIZE,
        transform=T.Compose(
            transforms=[
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply(
                    p=0.6, transforms=[T.RandomAffine(degrees=(-10, 10), scale=(0.9, 1.1), translate=(0.1, 0.1))]
                ),
            ]
        ),
    )

    dm = ZippedDataModule(
        dm1=_dm_source,
        dm2=_dm_target,
        match_type="max",
        num_workers=8,
        train_batch_size=BATCH_SIZE,
        val_batch_size=BATCH_SIZE,
        test_batch_size=BATCH_SIZE,
    )
    dm.prepare_data()
    dm.setup("fit")

    # ############### TEST AND ADAPT #################

    if not args.skip_initial:
        test_without_adaptation(model=model, datamodule=dm, cfg=cfg, offline=args.offline)

    train_adapt(model=model, datamodule=dm, n_epochs=args.num_epochs, cfg=cfg, offline=args.offline)

    return


def test_without_adaptation(model: FloodnetMslModuleWrapper, datamodule: ZippedDataModule, cfg: dict, offline: bool):
    dm = deepcopy(datamodule)
    dm.prepare_data()
    dm.setup("test")
    dm.test_dataloader = dm.train_dataloader

    if not offline:
        run_name = f"t_initial_floodnet_{cfg['experiment_name']}"
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
    )

    trainer.test(model=model, datamodule=dm)


def train_adapt(model: FloodnetMslModuleWrapper, datamodule: ZippedDataModule, cfg: dict, offline: bool, n_epochs: int):
    experiment_name = f"t_floodnet_{cfg['experiment_name']}"

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
                + "-{epoch:02d}-{step:03d}-{challenge_score_target:.4f}-best-challenge-floodnet",
                save_last=False,
            ),
        ],
        logger=wandb_logger if not offline else None,
    )

    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
