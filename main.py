import dotenv
import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf

from inz.data.data_module import XBDDataModule
from inz.models.util import get_loc_cls_weights, get_wandb_logger


@hydra.main(config_path="config", config_name="common", version_base="1.3")
def main(cfg: DictConfig) -> pl.Trainer:
    config = OmegaConf.to_container(cfg, resolve=True, enum_to_str=False)

    dotenv.load_dotenv()
    pl.seed_everything(config["seed"])
    torch.set_float32_matmul_precision(config["float32_matmul_precision"])
    device = torch.device("cuda")

    dm = hydra.utils.call(config["datamodule"]["datamodule"])
    assert isinstance(dm, XBDDataModule)
    dm.prepare_data()
    dm.setup("fit")

    print(f"{len(dm.train_dataloader())} train batches, {len(dm.val_dataloader())} val batches")

    # TODO support manually specified weights
    loc_weights, cls_weights = get_loc_cls_weights(
        dataloader=dm.train_dataloader(), device=device, drop_unclassified_class=True
    )
    print(f"Localization weights: {loc_weights}\nClassification weights: {cls_weights}")

    # todo optimizer from hydra
    model = hydra.utils.instantiate(config["module"]["module"])(class_weights=cls_weights).to(device)

    wandb_logger = get_wandb_logger(
        project=config["project_name"], watch_model=True, watch_model_log_frequency=400, watch_model_model=model
    )

    trainer = hydra.utils.instantiate(config["trainer"]["trainer"])(logger=wandb_logger)

    trainer.fit(model, datamodule=dm)

    return trainer


if __name__ == "__main__":
    main()
