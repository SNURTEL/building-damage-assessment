# Analysis of Aerial and Satellite Imagery for Emergency Scenarios

### Setup

Initialize submodules

```shell
git submodule update --init --remote
```

Install the project using [PDM](https://pdm-project.org/en/latest/)

```shell
pdm install
```

Copy the envfile and input your `wandb` API key

```shell
cp .env.sample .env
```

#### Datesets

- [xBD dataset](https://xview2.org/dataset) (51 GB download, 133 GB uncompressed, +69 GB pre-processed)

    - An account in the *xView2* challenge is required

- [FloodNet dataset](https://www.dropbox.com/scl/fo/k33qdif15ns2qv2jdxvhx/ANGaa8iPRhvlrvcKXjnmNRc?rlkey=ao2493wzl1cltonowjdbrnp7f&e=4&dl=0) (12 GB download, 13 GB uncompressed, +1 GB pre-processed)

Unpack the datasets and move / symlink them to `data/xBD` and `data/floodnet` respectively

Then, run preprocessing scripts, which will prepare the data for training:

```shell
pdm run scripts/make_data_xbd.py 512 data/xBD_processed_512 all
pdm run scripts/make_data_floodnet_rescuenet.py floodnet 512 data/rescuenet_processed_512 all
```

### Train

```shell
pdm run scripts/train.py \
    datamodule=<DATAMODULE> \
    module=<MODEL>
```

Default datamodule setup assumes a 24 GB VRAM GPU, override batch sizes if needed. Training using `tier1` and `tier3` xBD subsets converges after approximately 40 epochs and takes ~20h to train on an RTX 3090 using `BF16-mixed` precision.

### Adapt

```shell
pdm run scripts/adapt.py \
    [--events <EVENT1>,<EVENT2>,<EVENT_N>|--floodnet] \
    -d <DUMPED_HYDRA_CONFIG> \
    -c <MODEL_CHECKPOINT>
```

Find the hydra config directory (`.hydra`) and model checkpoint in output directory created during training (`outputs/experiment_name`). The dumped config will include everything you set during the training stage, including CLI overrides.

### Finetune

```shell
pdm run scripts/finetune.py \
    [--events <EVENT1>,<EVENT2>,<EVENT_N>|--floodnet] \
    -d <DUMPED_HYDRA_CONFIG> \
    -c <MODEL_CHECKPOINT>
```

### Evaluate

```shell
pdm run scripts/eval.py \
    [--events <EVENT1>,<EVENT2>,<EVENT_N>|--floodnet] \
    -d <DUMPED_HYDRA_CONFIG> \
    -c <MODEL_CHECKPOINT>
```
