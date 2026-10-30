"""The neuron lab (LAB.md E0, "The instrument"): a local page to see the cell and the networks, intervene, and compare
runs side by side. Every run is simulated by `sim.py`, so the page shows the one model, not a copy of it.

Run (either works):
    cd experiments && ../venv312/Scripts/python.exe -m dsnn.viz.server [--port 8770]
    venv312/Scripts/python.exe experiments/dsnn/viz/server.py [--port 8770]
Then open http://localhost:8770.
"""
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

if __package__ in (None, ""):                       # run as a script: make `dsnn` importable
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from dsnn import protocols as P
from dsnn.cell import MORPHOLOGY, Params
from dsnn.sim import simulate

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(os.path.dirname(HERE), "runs", "viz_calibration.json")

# the cell parameters the page lets you change: (name, label, min, max, step)
EDITABLE = [
    ("Cm", "capacitance Cm (µF/cm²)", 0.3, 1.5, 0.05),
    ("Rm", "membrane resistance Rm (Ω·cm²)", 5000, 50000, 1000),
    ("Ra", "axial resistivity Ra (Ω·cm)", 50, 400, 10),
    ("g_ampa", "AMPA per contact (nS)", 0.0, 3.0, 0.01),
    ("g_nmda", "NMDA per contact (nS)", 0.0, 4.0, 0.01),
    ("nmda_decay", "NMDA decay (ms)", 10, 200, 5),
    ("mg_gamma", "Mg block steepness γ (1/mV)", 0.03, 0.12, 0.002),
    ("dca_theta", "dCaAP threshold (mV)", -60, -20, 1),
    ("dca_w", "dCaAP current w (pA)", 0, 2000, 10),
    ("dca_tauK", "dCaAP decline τ_K", 0.05, 3.0, 0.05),
    ("dca_ref", "dCaAP refractory (ms)", 20, 500, 10),
    ("b", "adaptation jump b (pA)", 0, 300, 5),
    ("Vr", "reset V_r (mV)", -75, -45, 1),
]


def calibrated():
    """The rule-set cell and strengths (protocols.calibrated), cached on disk for the default Params."""
    key = repr(Params())
    try:
        c = json.load(open(CACHE))
        if c["key"] == key:
            return Params().with_(dca_w=c["cal"]["dca_w"], dca_Ith=c["cal"]["zone_rheobase"]), c["cal"]
    except (OSError, ValueError, KeyError):
        pass
    print("calibrating the cell by its rules (once, ~10 s) ...", flush=True)
    p, cal = P.calibrated(0.05)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    json.dump(dict(key=key, cal=cal), open(CACHE, "w"), indent=1)
    return p, cal


P0, CAL = calibrated()
SCENARIOS = P.scenarios(CAL)


def meta():
    return dict(
        morphology=[dict(name=n, parent=par, L=L, d=d, kind=k) for n, par, L, d, k in MORPHOLOGY],
        params={k: getattr(P0, k) for k in Params.names()},
        editable=[dict(name=n, label=l, lo=lo, hi=hi, step=s) for n, l, lo, hi, s in EDITABLE],
        calibration=CAL, scenarios=SCENARIOS)


def run(req):
    sc = next(s for s in SCENARIOS if s["key"] == req["scenario"])
    p = P0.with_(**{k: v for k, v in req.get("params", {}).items() if k in Params.names()})
    controls = {k: (v if isinstance(v, str) else float(v)) for k, v in req["controls"].items()}
    for k in ("A", "B", "use_pulse", "use_tuft", "cond", "test"):
        if k in controls:
            controls[k] = int(controls[k])
    if "seed" in controls:
        controls["seed"] = int(controls["seed"])
    st, wiring = P.build(sc["key"], controls)
    r = simulate([st], sc["T"], p, wiring=wiring, keep_syn=True)
    edges = wiring.edges if wiring else []
    return dict(t=np.round(r.t, 2).tolist(), V=np.round(r.V[0].astype(float), 1).tolist(),
                spikes=r.spikes[0], dcaap=r.dcaap[0], syn=r.syn[0], edges=edges,
                seconds=round(r.seconds, 2), steps=r.steps, cells=len(r.spikes[0]))


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, open(os.path.join(HERE, "index.html"), "rb").read(), "text/html; charset=utf-8")
        elif self.path == "/api/meta":
            self._send(200, meta())
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/api/run":
            return self._send(404, {"error": "not found"})
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            self._send(200, run(req))
        except Exception as e:                        # report the error to the page instead of dropping the request
            self._send(400, {"error": "%s: %s" % (type(e).__name__, e)})

    def log_message(self, fmt, *args):
        pass


def main(argv):
    port = int(argv[argv.index("--port") + 1]) if "--port" in argv else 8770
    print("the neuron lab: http://localhost:%d" % port, flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main(sys.argv[1:])
