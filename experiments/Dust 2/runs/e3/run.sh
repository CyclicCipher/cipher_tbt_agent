#!/bin/sh
# Experiment 3 training (EXPERIMENT_3.md). Compiled rerun path. Baseline reused from Experiment 1, not re-run.
PY=../../venv/Scripts/python.exe
S="emb o proj fc out"
sig() { for x in $S; do printf "sig.%s=%s " $x $1; done; }
run() { name=$1; shift; for s in 0 1; do
  $PY train.py --fast --seed $s --eval_every 10 --json runs/e3/m2/${name}_s$s.json "$@" > runs/e3/m2/${name}_s$s.log 2>&1
  tail -1 runs/e3/m2/${name}_s$s.log; done; }
run T1_sig01 --method dust --steps 300 --lr 1e-2 --set K=32 amp=fp16 $(sig 0.1)
run T1_sig04 --method dust --steps 300 --lr 1e-2 --set K=32 amp=fp16 $(sig 0.4)
run T1_anneal --method dust --steps 300 --lr 1e-2 --sig_sched 0.4,0.05 --set K=32 amp=fp16
run T2_boltz1 --method dust --steps 300 --lr 1e-2 --set K=32 amp=fp16 shaping=boltz lam=1.0
run T2_boltz03 --method dust --steps 300 --lr 1e-2 --set K=32 amp=fp16 shaping=boltz lam=0.3
run T2_rank --method dust --steps 300 --lr 1e-2 --set K=32 amp=fp16 shaping=rank
run T3_simul_K143 --method dust --steps 300 --lr 1e-2 --set K=143 amp=fp16 simul=1 sig.proj=0.1
echo ALLDONE
