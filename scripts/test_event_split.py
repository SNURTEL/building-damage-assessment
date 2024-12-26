import argparse
import itertools
import os
import re
import shlex
import subprocess
import sys
from enum import Enum
from pathlib import Path
from typing import Literal, NamedTuple

from rich.console import Console
from rich.pretty import pprint
from rich.text import Text

if Path.cwd().stem == "scripts":
    PROJECT_DIR = Path.cwd().parent
    os.chdir("..")
else:
    PROJECT_DIR = Path.cwd()

sys.path.append(str(PROJECT_DIR))

from inz.data.event import Event

EVENT_SPLIT_CONFIGS_DIR = PROJECT_DIR / "config" / "datamodule" / "event_split"
OUTPUTS_DIR = PROJECT_DIR / "outputs"


class ModelName(Enum):
    baseline = "baseline"
    farseg = "farseg"
    dahitra = "dahitra"


MODEL_NAMES = [mn.value for mn in ModelName]

EVENT_NAMES = [e.value.replace("-", "_") for e in Event]


class EventType(Enum):
    wind = "wind"
    flood = "flood"
    wildfire = "wildfire"


experiment_t = Literal["noadapt", "msl", "finetune"]


class TestConfig(NamedTuple):
    event_type: EventType
    test_event: Event
    model_name: ModelName
    datamodule_config_path: Path
    ckpt_path: Path
    hydra_config_dir: Path
    val_challenge_score: float


def get_test_configs(configs_path: Path) -> list[TestConfig]:
    configs = []

    for dm_cfg_file in configs_path.rglob("./test_*.yaml"):
        event_name = re.search(r"test_(.*)", dm_cfg_file.stem).group(1)
        if event_name not in EVENT_NAMES:
            continue

        event_type = dm_cfg_file.parent.stem
        models_output_paths = [
            (model_name, OUTPUTS_DIR / f"split_{event_type}_test_{event_name}_{model_name}")
            for model_name in MODEL_NAMES
        ]

        for model_name, output_path in models_output_paths:
            checkpoints_path = output_path / "latest_run" / "checkpoints"
            hydra_config_dir = output_path / "latest_run" / ".hydra"
            challenge_score_checkpoints = checkpoints_path.rglob("./*-challenge_score_safe-*.ckpt")
            ckpt_paths_with_score = [
                (float(re.search(r".*-challenge_score_safe-(0.\d+)-.*", path.stem).group(1)), path)
                for path in challenge_score_checkpoints
            ]
            score, ckpt_path = max(ckpt_paths_with_score, key=lambda sp: sp[0])

            configs.append(
                TestConfig(
                    event_type=EventType(event_type),
                    test_event=Event(event_name.replace("_", "-")),
                    model_name=ModelName(model_name),
                    datamodule_config_path=dm_cfg_file.resolve(),
                    ckpt_path=ckpt_path.resolve(),
                    hydra_config_dir=hydra_config_dir.resolve(),
                    val_challenge_score=score,
                )
            )

    return configs


def test_event(cfg: TestConfig, experiment: experiment_t, offline=False) -> None:
    if experiment == "noadapt":
        cmd = (
            f"pdm run python3 test.py -d {cfg.hydra_config_dir.relative_to(PROJECT_DIR)} -c {cfg.ckpt_path} -e {cfg.test_event.value}"
            + (" --offline" if offline else "")
        )
    elif experiment == "msl":
        cmd = (
            f"pdm run python3 scripts/train_test_msl.py --skip-initial -n 1 -d {'..' / cfg.hydra_config_dir.relative_to(PROJECT_DIR)} -c {cfg.ckpt_path} -e {cfg.test_event.value}"
            + (" --offline" if offline else "")
        )
    elif experiment == "finetune":
        # cmd = ""
        raise NotImplementedError(experiment)
    else:
        raise ValueError(f"{experiment} is not a valid experiment")
    pprint(f"Running cmd: {cmd}")
    with subprocess.Popen(shlex.split(cmd)) as handle:
        exit_code = handle.wait()
    print()
    if exit_code != 0:
        raise RuntimeError(f'Command "{cmd}" exited with error code {exit_code}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["noadapt", "msl", "finetune"], nargs=1, help="Experiment to run", required=True)
    parser.add_argument("--offline", action=argparse.BooleanOptionalAction, help="Do not log to wandb", default=False)
    parser.add_argument("-s", "--skip-n-first", type=int, help="Skip first N configs (useful for resuming)", default=0)
    args = parser.parse_args()
    test_configs = list(
        itertools.chain(
            *[get_test_configs(event_split_path) for event_split_path in EVENT_SPLIT_CONFIGS_DIR.glob("./*")]
        )
    )
    n_configs = len(test_configs)

    experiment = args.experiment[0]
    console = Console()
    text = Text.assemble((f"Running experiment {experiment.upper()}", "bold magenta"),)
    console.print(text)

    print(f"Running {max(n_configs - args.skip_n_first, 0)} test configs ({n_configs} total, {args.skip_n_first} skipped)")
    for i, cfg in enumerate(test_configs, start=1):
        if i <= args.skip_n_first:
            continue
        pprint(f"[{i} / {n_configs}] Running test with config:")
        pprint(cfg)
        test_event(cfg, experiment=experiment, offline=args.offline)
