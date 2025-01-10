#!/usr/bin/env bash

set -o xtrace

export HYDRA_FULL_ERROR=1

EXPERIMENT_NAME_BASE=split_wildfire_test


# for event in santa_rosa_wildfire portugal_wildfire woolsey_fire pinery_bushfire; do
for event in socal_fire; do
    DATAMODULE=event_split/wildfire/test_${event}

    # pdm run main.py datamodule=$DATAMODULE module=dahitra/dahitra_custom_weights +experiment_name=${EXPERIMENT_NAME_BASE}_${event}_dahitra datamodule.datamodule.train_batch_size=32 datamodule.datamodule.val_batch_size=32 datamodule.datamodule.test_batch_size=32 trainer.trainer.max_epochs=40 resume_from_checkpoint=;

#    pdm run main.py datamodule=$DATAMODULE module=baseline/baseline_manual_weights +experiment_name=${EXPERIMENT_NAME_BASE}_${event}_baseline datamodule.datamodule.train_batch_size=36 datamodule.datamodule.val_batch_size=36 datamodule.datamodule.test_batch_size=36 trainer.trainer.max_epochs=40 resume_from_checkpoint=;

    pdm run main.py datamodule=$DATAMODULE module=farseg/farseg_default_weights +experiment_name=${EXPERIMENT_NAME_BASE}_${event}_farseg datamodule.datamodule.train_batch_size=30 datamodule.datamodule.val_batch_size=30 datamodule.datamodule.test_batch_size=30 trainer.trainer.max_epochs=40 module.module.optimizer_factory.weight_decay=0.00001 resume_from_checkpoint=;
done
