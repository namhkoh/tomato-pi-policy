#!/usr/bin/env bash
# Learning curve on v1 (query-conditioned task): LoRA on plant-stratified target subsets, equal step budget, same validation eval + probe.
set -uo pipefail
cd /workspace/nhkoh/tomato-vlm/greenhouse_training_code/examples/greenhouse_sim
export VLM_DATA=/workspace/nhkoh/tomato-vlm/grounding_release VLM_MODEL=/workspace/nhkoh/tomato-vlm/models/Qwen3-VL-8B-Instruct VLM_RUNS=/workspace/nhkoh/tomato-vlm/runs
export WANDB_DIR=$VLM_RUNS/wandb HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY=/root/venvs/deleaf-qwen/bin/python; TR="/root/venvs/deleaf-qwen/bin/torchrun --standalone --nproc_per_node=4 -m sim_data.h200_train"
WB="--wandb-project tomato-pi --wandb-entity namhokoh-korea-advanced-institute-of-science-and-technology"
LOG=$VLM_RUNS/logs/chain-learning-curve.log; stage(){ echo "[$(date)] $*" >> "$LOG"; }
for sub in t010 t020 t040 t080 t168 all; do
  name=lc-$sub; ids=$VLM_RUNS/learning_curve/subset_$sub.json
  stage "$name train start"
  CUDA_VISIBLE_DEVICES=0,1,2,3 $TR --dataset $VLM_DATA --model $VLM_MODEL --output $VLM_RUNS/$name --mode train --max-steps 240 --gradient-accumulation 8 \
    --learning-rate 1e-4 --coordinate-decimals 2 --class-balance --status-token-weight 3 --train-ids $ids --run-name $name $WB > $VLM_RUNS/logs/$name.log 2>&1 \
    && stage "LC_${sub}_TRAIN_OK" || { stage "LC_${sub}_TRAIN_FAILED"; continue; }
  stage "$name eval start"
  bash $VLM_RUNS/eval-validation.sh $name-validation $VLM_RUNS/$name/adapter > $VLM_RUNS/logs/eval-$name.out 2>&1 && stage "LC_${sub}_EVAL_OK" || { stage "LC_${sub}_EVAL_FAILED"; continue; }
  for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i $PY -m sim_data.h200_status_probe probe --dataset $VLM_DATA --model $VLM_MODEL --adapter $VLM_RUNS/$name/adapter --split validation --output $VLM_RUNS/probe-$name-validation --shard-index $i --shard-count 4 > $VLM_RUNS/logs/probe-$name-$i.log 2>&1 & done; wait
  $PY -m sim_data.h200_status_probe summarize --run $VLM_RUNS/probe-$name-validation --output $VLM_RUNS/probe-$name-validation/summary.json > $VLM_RUNS/logs/probe-$name-summary.log 2>&1 && stage "LC_${sub}_PROBE_OK" || stage "LC_${sub}_PROBE_FAILED"
done
stage "LC_ALL_DONE"
