import importlib
import sys
from argparse import ArgumentParser

import dotenv
import hydra
import pytorch_lightning as pl
import torch
from hydra import compose, initialize
from pytorch_lightning.callbacks import RichProgressBar

from inz.data.event import Event, Hold

sys.path.append("inz/farseg")
sys.path.append("inz/dahitra")


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
        "-r", "--run-name", help="Run name; defaults to t_{original_run_name}", required=False, default=None
    )

    args = parser.parse_args()

    with initialize(version_base="1.3", config_path=args.hydra_config):
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

    BATCH_SIZE = cfg["datamodule"]["datamodule"]["train_batch_size"]

    dm = XBDDataModule(
        path=cfg["datamodule"]["datamodule"]["path"],
        drop_unclassified_channel=True,
        events={
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
        },
        val_fraction=0.0,
        test_fraction=1.0,
        train_batch_size=BATCH_SIZE,
        val_batch_size=BATCH_SIZE,
        test_batch_size=BATCH_SIZE,
    )
    dm.prepare_data()
    dm.setup("test")

    print(f"{len(dm.test_dataloader())} test batches")

    wandb_logger = get_wandb_logger(
        run_name=f"t_{cfg['experiment_name']}",
        project=cfg["project_name"],
        watch_model_model=model,
        dir="outputs/.wandb_tests",
    )

    wandb_logger.experiment.config["hydra_cfg"] = cfg

    trainer = pl.Trainer(max_epochs=1, callbacks=[RichProgressBar()], precision="bf16", logger=wandb_logger)
    trainer.test(model, datamodule=dm)

    return trainer


if __name__ == "__main__":
    main()
