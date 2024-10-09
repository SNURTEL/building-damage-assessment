import datetime
import importlib
import json
import os
import random
import string
from pathlib import Path

import dotenv
import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint

from inz.util import get_loc_cls_weights, get_wandb_logger, nested_dict_to_tuples


def get_cwd() -> Path:
    return Path(__file__).parent.resolve()


@hydra.main(config_path="config", config_name="common", version_base="1.3")
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

    if config["module"].get("no_weights") is not True:
        WEIGHT_CACHE_FILE = "class_weight_cache.json"
        try:
            with open(WEIGHT_CACHE_FILE, mode="r", encoding="utf-8") as fp:
                weight_cache = json.load(fp)
        except (FileNotFoundError, json.decoder.JSONDecodeError):
            weight_cache = {}

        key = str(
            nested_dict_to_tuples(
                config["datamodule"]["datamodule"].get("events") or config["datamodule"]["datamodule"].get("split_events")
            )
        )
        print(key)
        if hit := weight_cache.get(key):
            print("Found matching class weights in cache")
            loc_weights = torch.Tensor(hit["loc"]).to(device)
            cls_weights = torch.Tensor(hit["cls"]).to(device)
        else:
            print("Class weights not found in cache")
            loc_weights, cls_weights = get_loc_cls_weights(
                dataloader=dm.train_dataloader(), device=device, drop_unclassified_class=True
            )
            weight_cache[key] = {"loc": loc_weights.tolist(), "cls": cls_weights.tolist()}
            with open(WEIGHT_CACHE_FILE, mode="w", encoding="utf-8") as fp:
                weight_cache = json.dump(weight_cache, fp, indent=4)

        print(f"Localization weights: {loc_weights}\nClassification weights: {cls_weights}")
    elif config["module"].get("class_weights") is not None:
        cls_weights = hydra.initialize(config["module"]["class_weights"])
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
    else:
        model_ckpt = config.get("resume_from_checkpoint")

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
        run_name=f"{config['experiment_name']}-{datetime.datetime.now().replace(microsecond=0).isoformat()}-{''.join(random.choices(string.ascii_lowercase, k=8))}",
        project=config["project_name"],
        watch_model=True,
        watch_model_log_frequency=500,
        watch_model_model=model,
        dir=config["wandb_dir"],
    )

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
