#!/usr/bin/env bash

set -o xtrace

export HYDRA_FULL_ERROR=1


# adapt / finetune xbd to rescuenet

pdm run scripts/train_test_msl.py --rescuenet -d ../saved_checkpoints/runs/baseline_singlebranch/.hydra_lol -c /home/tomek/inz/inz/saved_checkpoints/runs/baseline_singlebranch/baseline_singlebranch_ckpt/baseline_singlebranch-epoch=33-step=16592-challenge_score_safe=0.639932-best-challenge-score.ckpt -n 10 --size min

pdm run scripts/train_test_msl.py --rescuenet -d ../saved_checkpoints/runs/farseg_single/latest_run/.hydra -c /home/tomek/inz/inz/saved_checkpoints/runs/farseg_single/latest_run/checkpoints/experiment_name-0-epoch-28-step-28275-challenge_score_safe-0.6489-best-challenge-score.ckpt -n 10 --size min

pdm run scripts/finetune_test.py --rescuenet -d ../saved_checkpoints/runs/baseline_singlebranch/.hydra_lol -c /home/tomek/inz/inz/saved_checkpoints/runs/baseline_singlebranch/baseline_singlebranch_ckpt/baseline_singlebranch-epoch=33-step=16592-challenge_score_safe=0.639932-best-challenge-score.ckpt -n 10 --skip-initial

pdm run scripts/finetune_test.py --rescuenet -d ../saved_checkpoints/runs/farseg_single/latest_run/.hydra -c /home/tomek/inz/inz/saved_checkpoints/runs/farseg_single/latest_run/checkpoints/experiment_name-0-epoch-28-step-28275-challenge_score_safe-0.6489-best-challenge-score.ckpt -n 10 --skip-initial


# adapt / finetune rescunet to floodnet

pdm run scripts/train_test_msl.py --rescuenet -d ../outputs/baseline_single_rescuenet/2024-12-31_04-36-19/.hydra -c /home/tomek/inz/inz/outputs/baseline_single_rescuenet/2024-12-31_04-36-19/checkpoints/experiment_name-0-epoch-26-step-3051-challenge_score_safe-0.8166-best-challenge-score.ckpt -n 10 --size min

pdm run scripts/train_test_msl.py --rescuenet -d ../outputs/farseg_single_rescuenet/2024-12-31_03-05-59/.hydra -c /home/tomek/inz/inz/outputs/farseg_single_rescuenet/2024-12-31_03-05-59/checkpoints/experiment_name-0-epoch-37-step-4294-challenge_score_safe-0.8180-best-challenge-score.ckpt -n 10 --size min

pdm run scripts/finetune_test.py --floodnet -d ../outputs/baseline_single_rescuenet/2024-12-31_04-36-19/.hydra -c /home/tomek/inz/inz/outputs/baseline_single_rescuenet/2024-12-31_04-36-19/checkpoints/experiment_name-0-epoch-26-step-3051-challenge_score_safe-0.8166-best-challenge-score.ckpt -n 10 --skip-initial

pdm run scripts/finetune_test.py --floodnet -d ../outputs/farseg_single_rescuenet/2024-12-31_03-05-59/.hydra -c /home/tomek/inz/inz/outputs/farseg_single_rescuenet/2024-12-31_03-05-59/checkpoints/experiment_name-0-epoch-37-step-4294-challenge_score_safe-0.8180-best-challenge-score.ckpt -n 10 --skip-initial
