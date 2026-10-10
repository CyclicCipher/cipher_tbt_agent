#!/bin/sh
# Experiment 2b training (EXPERIMENT_2B.md). Compiled rerun path (--fast). Comparisons are reused, not re-run.
PY=../../venv/Scripts/python.exe
run() { name=$1; shift; for s in 0 1; do
  $PY train.py --fast --seed $s --eval_every 10 --json runs/e2b/${name}_s$s.json "$@" > runs/e2b/${name}_s$s.log 2>&1
  tail -1 runs/e2b/${name}_s$s.log; done; }
run guided_w --method dust --steps 300 --lr 1e-2 --set K=32 amp=fp16 noise=guided rank=8 beta=0.5 whiten=1
run top_w --method dust --steps 300 --lr 1e-2 --set K=32 amp=fp16 top_guide=0.3 exact_head=1 whiten=1
run combo_w_K56 --method dust --steps 300 --lr 1e-2 --set K=56 amp=fp16 noise=guided rank=8 beta=0.5 sparse_c=2 top_guide=0.3 hub_T=1 exact_head=1 whiten=1
run O4_orth_K56 --method dust --steps 300 --lr 1e-2 --set K=56 amp=fp16 hub_T=1 noise=orth
run O4_local_K128 --method dust --steps 300 --lr 1e-2 --aux_every 1 --set K=128 amp=fp16 hub_T=1 local=1
echo ALLDONE
