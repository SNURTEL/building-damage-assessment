# inz

### Setup

##### 1. Install the project

```shell
pdm install
```

##### 2. Copy the envfile and input your `wandb` API key

```shell
cp .env.sample .env
```

##### 3. Get the [xBD dataset](https://xview2.org/dataset) and unpack (or symlink) it to `data/xBD`.

##### 4. Run the data preprocessing script

```shell
pdm run scripts/make_data.py
```

This will output the preprocessed data to `data/xBD_processed`


### Run sanity-check training:

```shell
pdm run main.py
```

This will use the hydra config from `config/common.yaml`.

Metrics will be logged to `wandb` (`inz` project).

Outputs will be written to `outputs/<EXPERIMENT_NAME>/<RUN_START_DATETIME>`, which will also be symlinked to `outputs/<EXPERIMENT_NAME>/latest_run`. Checkpoints can be found in the `checkpoints` subdirectory. NOTE - the symlinked dir updates with every run!

### Start an experiment

Simply replace the datamodule (or whatever component needs to be replaced):

```shell
pdm run main.py datamodule=<DATAMODULE_OVERRIDE>
```

e.g.

```shell
pdm run main.py datamodule=val_on_hurricanes 
```

### Resume a previous run

This will be tricky. Theoretically, if you specified `resume_from_checkpoint: last|if_exist`, the last run should resume automatically. If that does not happen for whatever reason, we'll have to override hydra's config with the one dumped when the previous run was created:

```shell
pdm run main.py --config-path <PREVIOUS_RUN_PATH>/.hydra --config-name config hydra.run.dir=<PREVIOUS_RUN_PATH> resume_from_checkpoint="<PREVIOUS_RUN_PATH>/checkpoints/<CHECKPOINT"
```

The run should be resumed from the checkpoint. The symlinked directory will NOT be updated.
