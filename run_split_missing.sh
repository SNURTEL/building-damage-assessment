#!/usr/bin/env bash

set -o xtrace

export HYDRA_FULL_ERROR=1

GLOBAL_BATCH_SIZE=30


EXPERIMENT_NAME_BASE=split_flood_test
event=hurricane_michael
DATAMODULE=event_split/flood/test_${event}
pdm run main.py datamodule=$DATAMODULE module=farseg/farseg_default_weights +experiment_name=${EXPERIMENT_NAME_BASE}_${event}_farseg datamodule.datamodule.train_batch_size=${GLOBAL_BATCH_SIZE} datamodule.datamodule.val_batch_size=${GLOBAL_BATCH_SIZE} datamodule.datamodule.test_batch_size=${GLOBAL_BATCH_SIZE} trainer.trainer.max_epochs=40 module.module.optimizer_factory.weight_decay=0.00001 resume_from_checkpoint=;


EXPERIMENT_NAME_BASE=split_wildfire_test
event=santa_rosa_wildfire
DATAMODULE=event_split/wildfire/test_${event}
pdm run main.py datamodule=$DATAMODULE module=dahitra/dahitra_custom_weights +experiment_name=${EXPERIMENT_NAME_BASE}_${event}_dahitra datamodule.datamodule.train_batch_size=${GLOBAL_BATCH_SIZE} datamodule.datamodule.val_batch_size=${GLOBAL_BATCH_SIZE} datamodule.datamodule.test_batch_size=${GLOBAL_BATCH_SIZE} trainer.trainer.max_epochs=40 resume_from_checkpoint=;
pdm run main.py datamodule=$DATAMODULE module=baseline/baseline_manual_weights +experiment_name=${EXPERIMENT_NAME_BASE}_${event}_baseline datamodule.datamodule.train_batch_size=${GLOBAL_BATCH_SIZE} datamodule.datamodule.val_batch_size=${GLOBAL_BATCH_SIZE} datamodule.datamodule.test_batch_size=${GLOBAL_BATCH_SIZE} trainer.trainer.max_epochs=40 resume_from_checkpoint=;
pdm run main.py datamodule=$DATAMODULE module=farseg/farseg_default_weights +experiment_name=${EXPERIMENT_NAME_BASE}_${event}_farseg datamodule.datamodule.train_batch_size=${GLOBAL_BATCH_SIZE} datamodule.datamodule.val_batch_size=${GLOBAL_BATCH_SIZE} datamodule.datamodule.test_batch_size=${GLOBAL_BATCH_SIZE} trainer.trainer.max_epochs=40 module.module.optimizer_factory.weight_decay=0.00001 resume_from_checkpoint=;


EXPERIMENT_NAME_BASE=split_wildfire_test
event=woolsey_fire
DATAMODULE=event_split/wildfire/test_${event}
pdm run main.py datamodule=$DATAMODULE module=dahitra/dahitra_custom_weights +experiment_name=${EXPERIMENT_NAME_BASE}_${event}_dahitra datamodule.datamodule.train_batch_size=${GLOBAL_BATCH_SIZE} datamodule.datamodule.val_batch_size=${GLOBAL_BATCH_SIZE} datamodule.datamodule.test_batch_size=${GLOBAL_BATCH_SIZE} trainer.trainer.max_epochs=40 resume_from_checkpoint=;
