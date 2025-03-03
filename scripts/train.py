import datetime
import importlib
import os
import sys
from pathlib import Path

import dotenv
import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint

from _ensure_cwd import ensure_cwd


PROJECT_DIR = ensure_cwd()

sys.path.append(PROJECT_DIR / "inz/external/farseg")
sys.path.append(PROJECT_DIR / "inz/external/dahitra")
sys.path.append(PROJECT_DIR / "inz/external/xview2_strong_baseline")

from inz.util import get_wandb_logger


def get_cwd() -> Path:
    return Path(__file__).parent.resolve()


@hydra.main(config_path=str(PROJECT_DIR / "config"), config_name="common", version_base="1.3")
def main(cfg: DictConfig) -> pl.Trainer:
    OmegaConf.register_new_resolver("cwd", get_cwd)
    config = OmegaConf.to_container(cfg, resolve=True, enum_to_str=False)

    dotenv.load_dotenv()
    pl.seed_everything(config["seed"])
    torch.set_float32_matmul_precision(config["float32_matmul_precision"])
    device = torch.device("cuda")

    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    last_job_symlink_dir = Path(config["last_job_symlink_dir"])

    dm = hydra.utils.call(config["datamodule"]["datamodule"])
    dm.prepare_data()
    dm.setup("fit")
    print(f"Loaded datamodule with {len(dm.train_dataloader())} train batches, {len(dm.val_dataloader())} val batches")

    if config["module"].get("class_weights") is not None:
        cls_weights = hydra.utils.instantiate(config["module"]["class_weights"]).to(device)
    else:
        cls_weights = None

    if config.get("resume_from_checkpoint") in ("last", "latest", "if-exists", "if_exists"):
        try:
            model_ckpt = next(
                (last_job_symlink_dir.resolve() / "checkpoints").resolve().rglob(f"*{config['last_ckpt_suffix']}*")
            )
            print(f"### Attempting to resume from checkpoint {model_ckpt} ###")
        except StopIteration:
            if config.get("resume_from_checkpoint") in ("if-exists", "if_exists"):
                print("### No checkpoint to resume from - starting a new run ###")
                model_ckpt = None
            else:
                print("### No checkpoint to resume from! ###")
                exit(1)
    elif config.get("resume_from_checkpoint"):
        model_ckpt = config.get("resume_from_checkpoint")
    else:
        model_ckpt = None

    if model_ckpt:
        model_class_str = config["module"]["module"]["_target_"]
        model_class_name = model_class_str.split(".")[-1]
        module_path = ".".join(model_class_str.split(".")[:-1])
        imported_module = importlib.import_module(module_path)
        model_class = getattr(imported_module, model_class_name)
        model_partial = hydra.utils.instantiate(config["module"]["module"])
        model = model_class.load_from_checkpoint(
            model_ckpt, *model_partial.args, **model_partial.keywords, class_weights=cls_weights
        )
    else:
        model = hydra.utils.instantiate(config["module"]["module"])(class_weights=cls_weights).to(device)

    model = model.to(device)

    try:
        # exclude re-running
        if not str(output_dir).endswith("latest_run"):
            last_job_symlink_dir.symlink_to(output_dir)
    except FileExistsError as e:
        if not last_job_symlink_dir.is_symlink:
            raise e
        os.unlink(last_job_symlink_dir)
        last_job_symlink_dir.symlink_to(output_dir)

    wandb_logger = get_wandb_logger(
        run_name=f"{config['experiment_name']}-{datetime.datetime.now().replace(microsecond=0).isoformat()}",
        project=config["project_name"],
        watch_model=True,
        watch_model_log_frequency=500,
        watch_model_model=model,
        dir=config["wandb_dir"],
    )

    wandb_logger.experiment.config["hydra_cfg"] = cfg
    wandb_logger.experiment.config["module_class"] = cfg["module"]["module"]["_target_"]
    wandb_logger.experiment.config["max_epochs"] = cfg["trainer"]["trainer"]["max_epochs"]
    wandb_logger.experiment.config["class_weights"] = cfg["module"].get("class_weights")
    wandb_logger.experiment.config["optimizer"] = cfg["module"]["module"].get("optimizer_factory")
    wandb_logger.experiment.config["scheduler"] = cfg["module"]["module"].get("scheduler_factory")
    wandb_logger.experiment.config["events_train"] = (
        cfg["datamodule"]["datamodule"].get("split_events", {}).get("train")
    )
    wandb_logger.experiment.config["events_val"] = cfg["datamodule"]["datamodule"].get("split_events", {}).get("val")
    wandb_logger.experiment.config["events_test"] = cfg["datamodule"]["datamodule"].get("split_events", {}).get("test")
    wandb_logger.experiment.config["events"] = cfg["datamodule"]["datamodule"].get("events")
    wandb_logger.experiment.config["val_fraction"] = cfg["datamodule"]["datamodule"].get("val_fraction")
    wandb_logger.experiment.config["val_test"] = cfg["datamodule"]["datamodule"].get("val_test")
    wandb_logger.experiment.config["train_batch_size"] = cfg["datamodule"]["datamodule"]["train_batch_size"]
    wandb_logger.experiment.config["val_batch_size"] = cfg["datamodule"]["datamodule"]["val_batch_size"]
    wandb_logger.experiment.config["test_batch_size"] = cfg["datamodule"]["datamodule"]["test_batch_size"]

    trainer = hydra.utils.instantiate(config["trainer"]["trainer"])(logger=wandb_logger)
    for callback in trainer.callbacks:
        if not isinstance(callback, ModelCheckpoint):
            continue
        # hydra will be very unhappy about using equal signs in file names - we need to replace them
        callback.CHECKPOINT_EQUALS_CHAR = "-"

    trainer.fit(model, datamodule=dm, ckpt_path=model_ckpt)

    return trainer


if __name__ == "__main__":
    main()
