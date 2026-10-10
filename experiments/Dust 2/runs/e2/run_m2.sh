#!/bin/sh
# Experiment 2 training. Run from the Dust 2 folder. Reused, NOT re-run: the Dust baseline (runs/e1/m2/base_s*),
# backprop seed 0 at 300 steps (2.1526, the lr check in EXPERIMENT_1.md §0). Every run logs val every 10 steps.
PY=../../venv/Scripts/python.exe
run() { dir=$1; name=$2; shift 2; for s in 0 1; do
  $PY train.py --seed $s --eval_every 10 --json runs/e2/$dir/${name}_s$s.json "$@" > runs/e2/$dir/${name}_s$s.log 2>&1
  tail -1 runs/e2/$dir/${name}_s$s.log; done; }
# V: variance or bias? 100 steps
run v dust_K32 --method dust --steps 100 --lr 1e-2 --set K=32 amp=fp16
run v dust_K128 --method dust --steps 100 --lr 1e-2 --set K=128 amp=fp16
run v bp --method bp --steps 100 --lr 2e-2
# M2: 300 steps, matched cost unless stated
run m2 O1_sparse --method dust --steps 300 --lr 1e-2 --set K=32 amp=fp16 sparse_c=2
run m2 O3_top --method dust --steps 300 --lr 1e-2 --set K=32 amp=fp16 top_guide=0.3 exact_head=1
run m2 O4_hubT_K56 --method dust --steps 300 --lr 1e-2 --set K=56 amp=fp16 hub_T=1
run m2 combo_K56 --method dust --steps 300 --lr 1e-2 --set K=56 amp=fp16 noise=guided rank=8 beta=0.5 sparse_c=2 top_guide=0.3 hub_T=1 exact_head=1
run m2 combo_local_K128 --method dust --steps 300 --lr 1e-2 --aux_every 1 --set K=128 amp=fp16 noise=guided rank=8 beta=0.5 sparse_c=2 top_guide=0.3 hub_T=1 exact_head=1 local=1
run m2 top10_base --method dust --steps 300 --lr 1e-2 --topk 0.1 --set K=32 amp=fp16
run m2 top10_O1 --method dust --steps 300 --lr 1e-2 --topk 0.1 --set K=32 amp=fp16 sparse_c=2
# bp_aux control (backprop seed 0 plain is reused; seed 1 plain is new)
run m2 bp_aux --method bp --steps 300 --lr 2e-2 --aux_every 1 --aux_weight 1.0
$PY train.py --seed 1 --eval_every 10 --json runs/e2/m2/bp_s1.json --method bp --steps 300 --lr 2e-2 > runs/e2/m2/bp_s1.log 2>&1; tail -1 runs/e2/m2/bp_s1.log
echo ALLDONE
