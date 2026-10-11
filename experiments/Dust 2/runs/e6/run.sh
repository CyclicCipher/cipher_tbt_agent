#!/bin/sh
# Experiment 6 Phase B: SGD + momentum 0.95. Adam comparisons reused, not re-run.
PY=../../venv/Scripts/python.exe
run() { name=$1; shift; for s in 0 1; do
  $PY train.py --fast --seed $s --eval_every 10 --json runs/e6/${name}_s$s.json "$@" > runs/e6/${name}_s$s.log 2>&1
  tail -1 runs/e6/${name}_s$s.log; done; }
run dust_sgd --method dust --opt sgd --lr 0.1 --emb_mult 30 --steps 300 --set K=32 amp=fp16
run bp_sgd --method bp --opt sgd --lr 0.1 --emb_mult 300 --steps 300
run O4_local_sgd --method dust --opt sgd --lr 0.1 --emb_mult 30 --steps 300 --aux_every 1 --set K=128 amp=fp16 hub_T=1 local=1
echo ALLDONE
