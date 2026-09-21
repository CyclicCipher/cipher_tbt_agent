cd "C:/Users/julia/GitHub/predictive-coding-agent/experiments/transformers"
export PYTHONIOENCODING=utf-8
../../venv/Scripts/python.exe h1_lid.py --pos rope --res std --steps 3200 --json ../ziplearn/runs/e8b/rope_std.json --save ../ziplearn/runs/e8b/rope_std.pt > ../ziplearn/runs/e8b/rope_std.log 2>&1
echo "rope_std done rc=$?"
../../venv/Scripts/python.exe h1_lid.py --pos rope --res attnres --steps 3200 --json ../ziplearn/runs/e8b/rope_attnres.json --save ../ziplearn/runs/e8b/rope_attnres.pt > ../ziplearn/runs/e8b/rope_attnres.log 2>&1
echo "rope_attnres done rc=$?"
