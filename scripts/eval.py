import argparse
import importlib
from pathlib import Path
import sys
from argparse import ArgumentParser
from pprint import pprint

import dotenv
import hydra
import pytorch_lightning as pl
import torch
from hydra import compose, initialize
from pytorch_lightning.callbacks import RichProgressBar

from _ensure_cwd import ensure_cwd

PROJECT_DIR = ensure_cwd()

from inz.data.data_module_frnet import FRNetModule
from inz.data.event import Event, Hold, Test, Tier1, Tier3

sys.path.append("inz/external/farseg")
sys.path.append("inz/external/dahitra")
sys.path.append("inz/external/dahitra/xBD_code")
sys.path.append("inz/external/xview2_strong_baseline")


from inz.data.data_module import XBDDataModule
from inz.util import get_wandb_logger


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
        "-e",
        "--events",
        help='Events to test on; either "hold" (all hold events) or a comma-separated list of events',
        required=True,
    )
    parser.add_argument(
        "-f", "--floodnet", action=argparse.BooleanOptionalAction, help="Use the floodnet dataset", default=False
    )
    parser.add_argument(
        "-r", "--run-name", help="Run name; defaults to t_{original_run_name}", required=False, default=None
    )
    parser.add_argument("--offline", action=argparse.BooleanOptionalAction, help="Do not log to wandb", default=False)

    args = parser.parse_args()

    assert (
        bool(args.events) + bool(args.floodnet) == 1
    ), "Provide exactly one of (--events, --floodnet)"

    with initialize(version_base="1.3", config_path=args.hydra_config):
        print(PROJECT_DIR)
        cfg = compose(config_name="config", overrides=[])

    model_class_str = cfg["module"]["module"]["_target_"]
    model_class_name = model_class_str.split(".")[-1]
    module_path = ".".join(model_class_str.split(".")[:-1])
    imported_module = importlib.import_module(module_path)
    model_class = getattr(imported_module, model_class_name)
    model_partial = hydra.utils.instantiate(cfg["module"]["module"])

    model = model_class.load_from_checkpoint(args.checkpoint_path, *model_partial.args, **model_partial.keywords).to(
        device
    )
    model.class_weights = model.class_weights.to(device)
    model.eval()

    BATCH_SIZE = cfg["datamodule"]["datamodule"]["train_batch_size"]

    if args.events == "hold":
        events = {
            Hold: [
                Event.guatemala_volcano,
                Event.hurricane_florence,
                Event.hurricane_harvey,
                Event.hurricane_matthew,
                Event.hurricane_michael,
                Event.mexico_earthquake,
                Event.midwest_flooding,
                Event.palu_tsunami,
                Event.santa_rosa_wildfire,
                Event.socal_fire,
            ],
        }
    else:
        events_l = set(args.events.replace("_", "-").split(","))
        events = {Tier1: [], Tier3: [], Test: [], Hold: []}
        for event_name in events_l:
            event = Event(event_name)
            for split in (Tier1, Tier3, Test, Hold):
                if event in split.events:
                    events[split].append(event)

    pprint(events)
    if args.floodnet:
        FRNetModule(
            path=Path(PROJECT_DIR / "data/floodnet_processed_512/FloodNet-Supervised_v1.0"),
            train_batch_size=BATCH_SIZE,
            val_batch_size=BATCH_SIZE,
            test_batch_size=BATCH_SIZE,
        )
    else:
        dm = XBDDataModule(
            path=PROJECT_DIR / "data/xBD_processed_512",
            drop_unclassified_channel=True,
            events=events,
            val_fraction=0.0,
            test_fraction=1.0,
            train_batch_size=BATCH_SIZE,
            val_batch_size=BATCH_SIZE,
            test_batch_size=BATCH_SIZE,
        )
    dm.prepare_data()
    dm.setup("test")

    print(f"{len(dm.test_dataloader())} test batches")

    if not args.offline:
        wandb_logger = get_wandb_logger(
            run_name=f"t_{cfg['experiment_name']}",
            project=cfg["project_name"],
            watch_model_model=model,
            dir="outputs/.wandb_tests",
        )
        wandb_logger.experiment.config["hydra_cfg"] = cfg

    trainer = pl.Trainer(
        max_epochs=1,
        callbacks=[RichProgressBar()],
        precision="bf16-mixed",
        logger=wandb_logger if not args.offline else None,
    )
    trainer.test(model, datamodule=dm)

    return trainer


if __name__ == "__main__":
    main()
