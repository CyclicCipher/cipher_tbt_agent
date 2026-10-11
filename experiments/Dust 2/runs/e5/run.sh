#!/bin/sh
# Experiment 5 training (EXPERIMENT_5.md). Compiled rerun path. Comparisons reused, not re-run.
PY=../../venv/Scripts/python.exe
run() { name=$1; shift; for s in 0 1; do
  $PY train.py --fast --seed $s --eval_every 10 --json runs/e5/${name}_s$s.json "$@" > runs/e5/${name}_s$s.log 2>&1
  tail -1 runs/e5/${name}_s$s.log; done; }
run W1_O4_local_orth_K128 --method dust --steps 300 --lr 1e-2 --aux_every 1 --set K=128 amp=fp16 hub_T=1 local=1 noise=orth_sign
run W2_O4_local_K192 --method dust --steps 300 --lr 1e-2 --aux_every 1 --set K=192 amp=fp16 hub_T=1 local=1
run W3_simul_local_K143 --method dust --steps 300 --lr 1e-2 --aux_every 1 --set K=143 amp=fp16 simul=1 local=1 sig.proj=0.1
echo ALLDONE
