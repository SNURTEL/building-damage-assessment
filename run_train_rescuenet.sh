#!/usr/bin/env bash

export HYDRA_FULL_ERROR=1

pdm run main.py datamodule=rescuenet.yaml module=farseg/farseg_singlebranch_default_weights +experiment_name=farseg_single_rescuenet datamodule.datamodule.train_batch_size=32 datamodule.datamodule.val_batch_size=32 datamodule.datamodule.test_batch_size=32 trainer.trainer.max_epochs=40 resume_from_checkpoint=

pdm run main.py datamodule=rescuenet.yaml module=baseline/baseline_singlebranch +experiment_name=baseline_single_rescuenet datamodule.datamodule.train_batch_size=32 datamodule.datamodule.val_batch_size=32 datamodule.datamodule.test_batch_size=32 trainer.trainer.max_epochs=40 resume_from_checkpoint=
