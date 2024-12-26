#!/usr/bin/env bash

set -x

N_EPOCHS=5

pdm run /home/tomek/inz/inz/scripts/train_test_msl.py --skip-initial -n $N_EPOCHS -d ../outputs/split_wind_test_hurricane_michael_baseline/2024-11-06_18-32-59/.hydra -c /home/tomek/inz/inz/outputs/split_wind_test_hurricane_michael_baseline/2024-11-06_18-32-59/checkpoints/experiment_name-0-epoch-18-step-2033-challenge_score_safe-0.6884-best-challenge-score.ckpt -e hurricane-michael

pdm run /home/tomek/inz/inz/scripts/train_test_msl.py --skip-initial -n $N_EPOCHS -d ../outputs/split_wind_test_hurricane_michael_farseg/2024-11-12_04-18-07/.hydra -c /home/tomek/inz/inz/outputs/split_wind_test_hurricane_michael_farseg/2024-11-12_04-18-07/checkpoints/experiment_name-0-epoch-33-step-4080-challenge_score_safe-0.7483-best-challenge-score.ckpt -e hurricane-michael

pdm run /home/tomek/inz/inz/scripts/train_test_msl.py --skip-initial -n $N_EPOCHS -d ../outputs/split_wind_test_hurricane_michael_dahitra/2024-11-06_14-51-55/.hydra -c /home/tomek/inz/inz/outputs/split_wind_test_hurricane_michael_dahitra/2024-11-06_14-51-55/checkpoints/experiment_name-0-epoch-42-step-5160-challenge_score_safe-0.7168-best-challenge-score.ckpt -e hurricane-michael

pdm run /home/tomek/inz/inz/scripts/train_test_msl.py --skip-initial -n $N_EPOCHS -d ../outputs/split_wind_test_moore_tornado_baseline/2024-11-07_11-32-25/.hydra -c /home/tomek/inz/inz/outputs/split_wind_test_moore_tornado_baseline/2024-11-07_11-32-25/checkpoints/experiment_name-0-epoch-27-step-3836-challenge_score_safe-0.6730-best-challenge-score.ckpt -e moore-tornado

pdm run /home/tomek/inz/inz/scripts/train_test_msl.py --skip-initial -n $N_EPOCHS -d ../outputs/split_wind_test_moore_tornado_farseg/2024-11-07_14-44-06/.hydra -c /home/tomek/inz/inz/outputs/split_wind_test_moore_tornado_farseg/2024-11-07_14-44-06/checkpoints/experiment_name-0-epoch-37-step-5852-challenge_score_safe-0.7016-best-challenge-score.ckpt -e moore-tornado

pdm run /home/tomek/inz/inz/scripts/train_test_msl.py --skip-initial -n $N_EPOCHS -d ../outputs/split_wind_test_moore_tornado_dahitra/2024-11-07_09-11-28/.hydra -c /home/tomek/inz/inz/outputs/split_wind_test_moore_tornado_dahitra/2024-11-07_09-11-28/checkpoints/experiment_name-0-epoch-39-step-6160-challenge_score_safe-0.6570-best-challenge-score.ckpt -e moore-tornado

pdm run /home/tomek/inz/inz/scripts/train_test_msl.py --skip-initial -n $N_EPOCHS -d ../outputs/split_wind_test_joplin_tornado_baseline/2024-11-07_03-16-48/.hydra -c /home/tomek/inz/inz/outputs/split_wind_test_joplin_tornado_baseline/2024-11-07_03-16-48/checkpoints/experiment_name-0-epoch-38-step-5655-challenge_score_safe-0.6394-best-challenge-score.ckpt -e joplin-tornado

pdm run /home/tomek/inz/inz/scripts/train_test_msl.py --skip-initial -n $N_EPOCHS -d ../outputs/split_wind_test_joplin_tornado_farseg/2024-11-07_06-41-53/.hydra -c /home/tomek/inz/inz/outputs/split_wind_test_joplin_tornado_farseg/2024-11-07_06-41-53/checkpoints/experiment_name-0-epoch-27-step-4564-challenge_score_safe-0.6874-best-challenge-score.ckpt -e joplin-tornado

pdm run /home/tomek/inz/inz/scripts/train_test_msl.py --skip-initial -n $N_EPOCHS -d ../outputs/split_wind_test_joplin_tornado_dahitra/2024-11-07_00-48-41/.hydra -c /home/tomek/inz/inz/outputs/split_wind_test_joplin_tornado_dahitra/2024-11-07_00-48-41/checkpoints/experiment_name-0-epoch-28-step-4727-challenge_score_safe-0.6417-best-challenge-score.ckpt -e joplin-tornado

pdm run /home/tomek/inz/inz/scripts/train_test_msl.py --skip-initial -n $N_EPOCHS -d ../outputs/split_wind_test_tuscaloosa_tornado_baseline/2024-11-07_23-50-52/.hydra -c /home/tomek/inz/inz/outputs/split_wind_test_tuscaloosa_tornado_baseline/2024-11-07_23-50-52/checkpoints/experiment_name-0-epoch-31-step-4032-challenge_score_safe-0.6892-best-challenge-score.ckpt -e tuscaloosa-tornado

pdm run /home/tomek/inz/inz/scripts/train_test_msl.py --skip-initial -n $N_EPOCHS -d ../outputs/split_wind_test_tuscaloosa_tornado_farseg/2024-11-12_02-07-44/.hydra -c /home/tomek/inz/inz/outputs/split_wind_test_tuscaloosa_tornado_farseg/2024-11-12_02-07-44/checkpoints/experiment_name-0-epoch-25-step-3692-challenge_score_safe-0.7232-best-challenge-score.ckpt -e tuscaloosa-tornado

pdm run /home/tomek/inz/inz/scripts/train_test_msl.py --skip-initial -n $N_EPOCHS -d ../outputs/split_wind_test_tuscaloosa_tornado_dahitra/2024-11-07_17-06-19/.hydra -c /home/tomek/inz/inz/outputs/split_wind_test_tuscaloosa_tornado_dahitra/2024-11-07_17-06-19/checkpoints/experiment_name-0-epoch-36-step-5254-challenge_score_safe-0.6793-best-challenge-score.ckpt -e tuscaloosa-tornado
