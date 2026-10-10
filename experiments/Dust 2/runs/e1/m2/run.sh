#!/bin/sh
# Experiment 1 M2: 300 steps, 2 seeds, matched cost. Run from the Dust 2 folder.
PY=../../venv/Scripts/python.exe
run() { name=$1; shift; for s in 0 1; do
  $PY train.py --method dust --steps 300 --lr 1e-2 --eval_every 100 --seed $s --json runs/e1/m2/${name}_s$s.json "$@" > runs/e1/m2/${name}_s$s.log 2>&1
  tail -1 runs/e1/m2/${name}_s$s.log; done; }
run base --set K=32 amp=fp16
run lr_rand --set K=32 amp=fp16 noise=lr_rand rank=8
run lr_pca --set K=32 amp=fp16 noise=lr_pca rank=8
run orth --set K=32 amp=fp16 noise=orth
run sobol --set K=32 amp=fp16 noise=sobol
run anti --set K=32 amp=fp16 noise=anti
run local128 --aux_every 1 --set K=128 amp=fp16 local=1
run ent50 --set K=32 amp=fp16 ent_frac=0.5
run guided --set K=32 amp=fp16 noise=guided rank=8 beta=0.5
echo ALLDONE
