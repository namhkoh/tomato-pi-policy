#!/usr/bin/env bash
# hamster-01: fine-tune the released 3D HAMSTER checkpoint on the cut-point task (bridge experiment), then evaluate.
set -uo pipefail
cd /workspace/nhkoh/3D_HAMSTER
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_DIR=/workspace/nhkoh/tomato-vlm/runs/wandb TOKENIZERS_PARALLELISM=false
D=/workspace/nhkoh/tomato-vlm/grounding_release; CK=/workspace/nhkoh/3D_HAMSTER/ckpt; R=/workspace/nhkoh/tomato-vlm/runs
PY=/root/venvs/hamster3d/bin/python; TR="/root/venvs/hamster3d/bin/torchrun --standalone --nproc_per_node=4 agro/hamster_train.py"
WB="--wandb-project tomato-pi --wandb-entity namhokoh-korea-advanced-institute-of-science-and-technology"
COMMON="--dataset $D --model $CK --deepspeed $R/ds_zero2.json --learning-rate 1e-5 --vision-learning-rate 1e-6 --class-balance --status-token-weight 3 --longest-edge 640 $WB"
LOG=$R/logs/chain-hamster-01.log; mkdir -p $R/logs
stage(){ echo "[$(date)] $*" >> "$LOG"; }
stage "smoke start"
CUDA_VISIBLE_DEVICES=0,1,2,3 $TR $COMMON --output $R/hamster-01-smoke --mode smoke --run-name hamster-01-smoke > $R/logs/hamster-01-smoke.log 2>&1 && stage "H01_SMOKE_OK" || { stage "H01_SMOKE_FAILED"; exit 1; }
stage "overfit start"
CUDA_VISIBLE_DEVICES=0,1,2,3 $TR $COMMON --output $R/hamster-01-overfit --mode overfit --run-name hamster-01-overfit > $R/logs/hamster-01-overfit.log 2>&1 && stage "H01_OVERFIT_OK" || { stage "H01_OVERFIT_FAILED"; exit 1; }
stage "reload start"
CUDA_VISIBLE_DEVICES=3 $PY agro/hamster_evaluate.py generate --dataset $D --model $R/hamster-01-overfit/model --split train --ids-file $R/hamster-01-overfit/run_contract.json --output $R/hamster-01-overfit-reload > $R/logs/hamster-01-reload.log 2>&1 \
  && $PY agro/hamster_evaluate.py score --dataset $D --run $R/hamster-01-overfit-reload --output $R/hamster-01-overfit-reload/report.json >> $R/logs/hamster-01-reload.log 2>&1 \
  && stage "H01_RELOAD_OK" || { stage "H01_RELOAD_FAILED"; exit 1; }
stage "train start"
CUDA_VISIBLE_DEVICES=0,1,2,3 $TR $COMMON --output $R/hamster-01 --mode train --epochs 3 --gradient-accumulation 8 --run-name hamster-01 > $R/logs/hamster-01.log 2>&1 && stage "H01_TRAIN_OK" || { stage "H01_TRAIN_FAILED"; exit 1; }
stage "eval start"
for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i $PY agro/hamster_evaluate.py generate --dataset $D --model $R/hamster-01/model --output $R/eval-hamster-01-validation --shard-index $i --shard-count 4 > $R/logs/eval-hamster-01-shard$i.log 2>&1 & done; wait
$PY agro/hamster_evaluate.py score --dataset $D --run $R/eval-hamster-01-validation --output $R/eval-hamster-01-validation/report.json > $R/logs/eval-hamster-01-score.log 2>&1 && stage "H01_EVAL_OK" || { stage "H01_EVAL_FAILED"; exit 1; }
stage "probe start"
for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i $PY agro/hamster_evaluate.py probe --dataset $D --model $R/hamster-01/model --output $R/probe-hamster-01-validation --shard-index $i --shard-count 4 > $R/logs/probe-hamster-01-shard$i.log 2>&1 & done; wait
$PY agro/hamster_evaluate.py summarize-probe --run $R/probe-hamster-01-validation --output $R/probe-hamster-01-validation/summary.json > $R/logs/probe-hamster-01-summary.log 2>&1 && stage "H01_PROBE_OK" || stage "H01_PROBE_FAILED"
stage "H01_ALL_DONE"
