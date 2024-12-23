#!/usr/bin/env bash

COMMAND="pdm run python3 scripts/train_test_msl.py"
COMMON_PARAMS="-n 5 --skip-initial"


$COMMAND $COMMON_PARAMS -e hurricane_matthew \
    -d ../outputs/split_wind_test_hurricane_matthew_baseline/latest_run/.hydra \
    -c /home/tomek/inz/inz/outputs/split_wind_test_hurricane_matthew_baseline/latest_run/checkpoints/experiment_name-0-epoch-12-step-1560-challenge_score_safe-0.6650-best-challenge-score.ckpt

$COMMAND $COMMON_PARAMS -e hurricane_matthew \
    -d  \
    -c 