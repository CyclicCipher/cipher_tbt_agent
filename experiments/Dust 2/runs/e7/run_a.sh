#!/bin/sh
# Experiment 7a: Block AttnRes alone. Comparisons reused, not re-run.
PY=../../venv/Scripts/python.exe
run() { name=$1; shift; for s in 0 1; do
  $PY train.py --fast --attnres --seed $s --eval_every 10 --json runs/e7/${name}_s$s.json "$@" > runs/e7/${name}_s$s.log 2>&1
  tail -1 runs/e7/${name}_s$s.log; done; }
run bp_ar --method bp --steps 300 --lr 2e-2
run dust_ar_O4_K56 --method dust --steps 300 --lr 1e-2 --set K=56 amp=fp16 hub_T=1
echo ALLDONE
