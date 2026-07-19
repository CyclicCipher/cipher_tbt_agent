"""imagination_widget.py -- the two-pane IMAGINATION widget (`project_hippocampus_imagination_and_widget`).

Plays the pure-push game (`tasks/games/push.py`) with the agent and, at every step, records the frame beside the agent's
IMAGINED rollout (`Agent.imagine`, the last committed plan unrolled through the learned forward model). Emits ONE self-contained
HTML file: REALITY (left) beside IMAGINATION (right), a step scrubber, the discovered goal, the plan, and a live model-vs-reality
check -- where they diverge is where the model is wrong. On the go-around level the agent plans the whole push from step 0.

Run (from the repo root, venv active):  PYTHONPATH=src python -m demos.imagination_widget [out.html]
"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # so `tbt`/`tasks` import when run as a script
from tbt.agent import Agent
from tasks.games.push import Push
from tasks.harness import Environment

C_WALL, C_AGENT, C_BOX, C_PAD = 1, 2, 6, 7


def _bbox(grid):
    cells = [(x, y) for y in range(len(grid)) for x in range(len(grid[0])) if grid[y][x] != 0]
    xs = [c[0] for c in cells]; ys = [c[1] for c in cells]
    return min(xs), min(ys), max(xs), max(ys)


def capture(seed=0, budget=400):
    game = Push(); env = Environment(game); fd = env.reset()
    agent = Agent(feat_n=8, n_content=2, n_state=2, n_cols=64, seed=seed)
    levels = {}
    while not (fd.is_terminal() or fd.is_win()):
        lvl = fd.level
        action, _ = agent.step(fd)
        objs = agent.transduce(fd.grid); pos = agent._positions(objs)
        imagined = agent.imagine()
        x0, y0, x1, y1 = _bbox(fd.grid)
        rec = levels.setdefault(lvl, {"w": x1 - x0 + 1, "h": y1 - y0 + 1, "static": None, "steps": []})
        if rec["static"] is None:
            rec["static"] = [[fd.grid[y][x] if fd.grid[y][x] in (C_WALL, C_PAD) else 0
                              for x in range(x0, x1 + 1)] for y in range(y0, y1 + 1)]
        def rel(c):
            return [c[0] - x0, c[1] - y0] if c else None
        rec["steps"].append({
            "agent": rel(pos.get(C_AGENT)),
            "box": rel(pos.get(C_BOX)),
            "plan": [a.name for a in agent._last_plan],
            "imagined": [{"agent": rel(s["agent"]), "box": rel(s["objects"].get(C_BOX))} for s in imagined],
            "goal": list(agent.goal_mem.goal()) if isinstance(agent.goal_mem.goal(), tuple) else agent.goal_mem.goal(),
        })
        fd = env.step(action)
    return {"won": fd.is_win(), "seed": seed,
            "levels": [{"id": k, **v} for k, v in sorted(levels.items())]}


DATA = capture(seed=0)

ACTION_ARROW = {"ACTION1": "↑", "ACTION2": "↓", "ACTION3": "←", "ACTION4": "→"}

HTML = r"""<title>Agent Imagination — Reality vs. Rollout</title>
<style>
  :root {
    --bg:#0b0f14; --panel:#111821; --grid-line:#1b2733; --floor:#0e141b;
    --ink:#e6edf3; --dim:#8aa0b2; --faint:#3d4c5a;
    --wall:#33414f; --agent:#35c1e8; --box:#f0a331; --pad:#39d98a; --accent:#35c1e8;
    --ghost:rgba(53,193,232,.16); --trail:rgba(240,163,49,.28);
    --good:#39d98a; --warn:#f0a331;
    --mono:ui-monospace,"Cascadia Code","SF Mono",Menlo,Consolas,monospace;
    --sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  }
  @media (prefers-color-scheme: light){
    :root{ --bg:#f2f5f8; --panel:#ffffff; --grid-line:#e3e9ef; --floor:#f7fafc;
      --ink:#14202b; --dim:#5a6b78; --faint:#c2ced8;
      --wall:#b7c3cd; --agent:#0e8fb8; --box:#c9791a; --pad:#1a9e63; --accent:#0e8fb8;
      --ghost:rgba(14,143,184,.12); --trail:rgba(201,121,26,.26); }
  }
  :root[data-theme="dark"]{ --bg:#0b0f14; --panel:#111821; --grid-line:#1b2733; --floor:#0e141b;
    --ink:#e6edf3; --dim:#8aa0b2; --faint:#3d4c5a; --wall:#33414f; --agent:#35c1e8; --box:#f0a331;
    --pad:#39d98a; --accent:#35c1e8; --ghost:rgba(53,193,232,.16); --trail:rgba(240,163,49,.28); }
  :root[data-theme="light"]{ --bg:#f2f5f8; --panel:#ffffff; --grid-line:#e3e9ef; --floor:#f7fafc;
    --ink:#14202b; --dim:#5a6b78; --faint:#c2ced8; --wall:#b7c3cd; --agent:#0e8fb8; --box:#c9791a;
    --pad:#1a9e63; --accent:#0e8fb8; --ghost:rgba(14,143,184,.12); --trail:rgba(201,121,26,.26); }

  *{ box-sizing:border-box; }
  body{ margin:0; background:var(--bg); color:var(--ink); font-family:var(--sans);
    line-height:1.5; -webkit-font-smoothing:antialiased; }
  .wrap{ max-width:940px; margin:0 auto; padding:32px 20px 64px; }
  header{ border-bottom:1px solid var(--grid-line); padding-bottom:20px; margin-bottom:24px; }
  .eyebrow{ font-family:var(--mono); font-size:11px; letter-spacing:.18em; text-transform:uppercase;
    color:var(--accent); margin:0 0 8px; }
  h1{ font-size:26px; line-height:1.2; margin:0 0 10px; text-wrap:balance; font-weight:650; letter-spacing:-.01em; }
  .lede{ color:var(--dim); max-width:64ch; margin:0; font-size:15px; }
  .lede b{ color:var(--ink); font-weight:600; }

  .tabs{ display:flex; gap:8px; margin:22px 0 18px; }
  .tab{ font-family:var(--mono); font-size:12.5px; padding:7px 14px; border-radius:7px; cursor:pointer;
    background:var(--panel); color:var(--dim); border:1px solid var(--grid-line); transition:.15s; }
  .tab[aria-selected="true"]{ color:var(--ink); border-color:var(--accent); box-shadow:inset 0 0 0 1px var(--accent); }
  .tab:focus-visible{ outline:2px solid var(--accent); outline-offset:2px; }

  .panes{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }
  @media (max-width:640px){ .panes{ grid-template-columns:1fr; } }
  .pane{ background:var(--panel); border:1px solid var(--grid-line); border-radius:12px; padding:14px 14px 16px; }
  .pane h2{ font-family:var(--mono); font-size:11px; letter-spacing:.14em; text-transform:uppercase; margin:0 0 2px;
    display:flex; align-items:center; gap:8px; }
  .dot{ width:8px; height:8px; border-radius:50%; }
  .pane .sub{ color:var(--dim); font-size:12px; margin:0 0 12px; min-height:16px; font-family:var(--mono); }
  canvas{ width:100%; height:auto; display:block; image-rendering:auto; border-radius:6px; }

  .controls{ display:flex; align-items:center; gap:14px; margin:20px 0 8px; }
  button.play{ font-family:var(--mono); font-size:13px; background:var(--accent); color:var(--bg); border:0;
    padding:9px 16px; border-radius:8px; cursor:pointer; font-weight:600; min-width:92px; }
  button.play:focus-visible{ outline:2px solid var(--ink); outline-offset:2px; }
  input[type=range]{ flex:1; accent-color:var(--accent); }
  .tick{ font-family:var(--mono); font-size:12.5px; color:var(--dim); font-variant-numeric:tabular-nums; white-space:nowrap; }

  .readout{ display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-top:18px; }
  @media (max-width:640px){ .readout{ grid-template-columns:1fr; } }
  .card{ background:var(--panel); border:1px solid var(--grid-line); border-radius:10px; padding:12px 14px; }
  .card .k{ font-family:var(--mono); font-size:10px; letter-spacing:.14em; text-transform:uppercase; color:var(--dim); margin:0 0 6px; }
  .card .v{ font-family:var(--mono); font-size:14px; color:var(--ink); }
  .plan-seq{ font-size:20px; letter-spacing:3px; line-height:1; }
  .status{ display:inline-flex; align-items:center; gap:7px; font-weight:600; }
  .legend{ display:flex; flex-wrap:wrap; gap:16px; margin-top:22px; font-family:var(--mono); font-size:11.5px; color:var(--dim); }
  .legend span{ display:inline-flex; align-items:center; gap:6px; }
  .legend i{ width:11px; height:11px; border-radius:3px; display:inline-block; }
  .note{ color:var(--dim); font-size:12.5px; margin-top:20px; max-width:66ch; }
  @media (prefers-reduced-motion:reduce){ *{ transition:none!important; } }
</style>

<div class="wrap">
  <header>
    <p class="eyebrow">Thousand-Brains agent · hippocampal rollout</p>
    <h1>What the agent is thinking</h1>
    <p class="lede">The agent plans by <b>imagining</b> a rollout through the world-model it learned online — no rules coded.
      Left is <b>reality</b>; right is the <b>imagined future</b> it committed to. Where they track, the model is right; where
      they diverge, that's the bug. On the go-around level it plans the entire push from step&nbsp;0.</p>
  </header>

  <div class="tabs" role="tablist" id="tabs"></div>

  <div class="panes">
    <div class="pane">
      <h2><span class="dot" style="background:var(--agent)"></span>Reality</h2>
      <p class="sub" id="realSub"></p>
      <canvas id="real"></canvas>
    </div>
    <div class="pane">
      <h2><span class="dot" style="background:var(--accent)"></span>Imagination — the plan</h2>
      <p class="sub" id="imagSub"></p>
      <canvas id="imag"></canvas>
    </div>
  </div>

  <div class="controls">
    <button class="play" id="play">Play</button>
    <input type="range" id="scrub" min="0" value="0" step="1" aria-label="step">
    <span class="tick" id="tick"></span>
  </div>

  <div class="readout">
    <div class="card"><p class="k">Discovered goal</p><div class="v" id="goal">—</div></div>
    <div class="card"><p class="k">Plan from here</p><div class="v plan-seq" id="plan">—</div></div>
    <div class="card"><p class="k">Model vs. reality</p><div class="v" id="status">—</div></div>
  </div>

  <div class="legend">
    <span><i style="background:var(--agent)"></i>self</span>
    <span><i style="background:var(--box)"></i>block</span>
    <span><i style="background:var(--pad)"></i>pad (goal)</span>
    <span><i style="background:var(--wall)"></i>wall</span>
    <span><i style="background:var(--accent);opacity:.4"></i>imagined</span>
  </div>
  <p class="note">The relation <span style="font-family:var(--mono)">block-on-pad</span> was discovered from the sparse
    score on an earlier level and transfers here — nothing about the block's position carries over, only the relation.
    The imagined block-path (amber trail) is the rollout's plan; reality follows it.</p>
</div>

<script>
const DATA = __DATA__;
const css = k => getComputedStyle(document.documentElement).getPropertyValue(k).trim();
const ARROW = {ACTION1:"↑",ACTION2:"↓",ACTION3:"←",ACTION4:"→"};

let L = 0, T = 0, subT = 0, playing = false, timer = null;

const tabsEl=document.getElementById("tabs");
DATA.levels.forEach((lv,i)=>{
  const b=document.createElement("button"); b.className="tab"; b.role="tab";
  b.textContent = (lv.steps[lv.steps.length-1].goal ? "Go-around push" : "Explore & discover");
  b.setAttribute("aria-selected", i===0);
  b.onclick=()=>{ L=i; T=0; subT=0; syncTabs(); resize(); render(); };
  tabsEl.appendChild(b);
});
function syncTabs(){ [...tabsEl.children].forEach((b,i)=>b.setAttribute("aria-selected", i===L)); }

const realC=document.getElementById("real"), imagC=document.getElementById("imag");
const scrub=document.getElementById("scrub");

function grid(){ return DATA.levels[L]; }
function cell(){ const g=grid(); return Math.floor(Math.min(360/g.w, 300/g.h)); }

function resize(){
  const g=grid(), cs=cell(), dpr=window.devicePixelRatio||1;
  [realC,imagC].forEach(c=>{ c.width=g.w*cs*dpr; c.height=g.h*cs*dpr;
    c.style.width=(g.w*cs)+"px"; c.style.height=(g.h*cs)+"px";
    c.getContext("2d").setTransform(dpr,0,0,dpr,0,0); });
  scrub.max=grid().steps.length-1;
}

function roundRect(ctx,x,y,w,h,r){ ctx.beginPath(); ctx.roundRect(x,y,w,h,r); }

function drawBase(ctx){
  const g=grid(), cs=cell();
  ctx.clearRect(0,0,g.w*cs,g.h*cs);
  ctx.fillStyle=css("--floor"); ctx.fillRect(0,0,g.w*cs,g.h*cs);
  ctx.strokeStyle=css("--grid-line"); ctx.lineWidth=1;
  for(let x=0;x<=g.w;x++){ ctx.beginPath(); ctx.moveTo(x*cs+.5,0); ctx.lineTo(x*cs+.5,g.h*cs); ctx.stroke(); }
  for(let y=0;y<=g.h;y++){ ctx.beginPath(); ctx.moveTo(0,y*cs+.5); ctx.lineTo(g.w*cs,y*cs+.5); ctx.stroke(); }
  for(let y=0;y<g.h;y++)for(let x=0;x<g.w;x++){
    const v=g.static[y][x];
    if(v===1){ ctx.fillStyle=css("--wall"); ctx.fillRect(x*cs,y*cs,cs,cs); }
    else if(v===7){ ctx.strokeStyle=css("--pad"); ctx.lineWidth=2;
      ctx.beginPath(); ctx.arc(x*cs+cs/2,y*cs+cs/2,cs*0.28,0,7); ctx.stroke();
      ctx.fillStyle=css("--pad"); ctx.globalAlpha=.12; ctx.fillRect(x*cs,y*cs,cs,cs); ctx.globalAlpha=1; }
  }
}
function agentGlyph(ctx,c,col,alpha,dashed){
  const cs=cell(); ctx.globalAlpha=alpha;
  ctx.fillStyle=col; ctx.strokeStyle=col; ctx.lineWidth=2;
  ctx.beginPath(); ctx.arc(c[0]*cs+cs/2,c[1]*cs+cs/2,cs*0.30,0,7);
  if(dashed){ ctx.setLineDash([4,3]); ctx.stroke(); ctx.setLineDash([]); } else { ctx.fill(); }
  ctx.globalAlpha=1;
}
function boxGlyph(ctx,c,col,alpha,dashed){
  const cs=cell(), p=cs*0.16; ctx.globalAlpha=alpha;
  ctx.lineWidth=2; roundRect(ctx,c[0]*cs+p,c[1]*cs+p,cs-2*p,cs-2*p,4);
  if(dashed){ ctx.strokeStyle=col; ctx.setLineDash([4,3]); ctx.stroke(); ctx.setLineDash([]); }
  else { ctx.fillStyle=col; ctx.fill(); }
  ctx.globalAlpha=1;
}

function renderReal(){
  const ctx=realC.getContext("2d"), s=grid().steps[T];
  drawBase(ctx);
  if(s.box) boxGlyph(ctx,s.box,css("--box"),1,false);
  if(s.agent) agentGlyph(ctx,s.agent,css("--agent"),1,false);
}
function renderImag(){
  const ctx=imagC.getContext("2d"), s=grid().steps[T], cs=cell();
  drawBase(ctx);
  const traj=s.imagined;
  // amber trail of the imagined block path
  ctx.fillStyle=css("--trail");
  traj.forEach(f=>{ if(f.box){ ctx.beginPath(); ctx.arc(f.box[0]*cs+cs/2,f.box[1]*cs+cs/2,cs*0.10,0,7); ctx.fill(); } });
  // faint ghost of the imagined self path
  ctx.strokeStyle=css("--ghost"); ctx.lineWidth=cs*0.5; ctx.lineCap="round"; ctx.lineJoin="round";
  ctx.beginPath(); let started=false;
  traj.forEach(f=>{ if(f.agent){ const x=f.agent[0]*cs+cs/2,y=f.agent[1]*cs+cs/2;
    started?ctx.lineTo(x,y):ctx.moveTo(x,y); started=true; } });
  ctx.stroke();
  const f=traj[Math.min(subT,traj.length-1)];
  if(f.box) boxGlyph(ctx,f.box,css("--box"),0.9,true);
  if(f.agent) agentGlyph(ctx,f.agent,css("--accent"),0.9,true);
}

function diverges(){
  const st=grid().steps, s=st[T]; if(T+1>=st.length||s.imagined.length<2) return null;
  const pred=s.imagined[1], next=st[T+1];
  const eq=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
  return eq(pred.agent,next.agent) && eq(pred.box,next.box);
}

function render(){
  const g=grid(), s=g.steps[T];
  scrub.value=T;
  document.getElementById("tick").textContent=`step ${T+1} / ${g.steps.length}`;
  document.getElementById("realSub").textContent = s.goal ? "executing the plan" : "exploring — goal still hidden";
  document.getElementById("imagSub").textContent = s.plan.length ? `rollout · ${s.plan.length} steps ahead` : "no plan — acting on novelty";
  document.getElementById("goal").innerHTML = s.goal ? `block <b style="color:var(--box)">on</b> pad` : "not yet found";
  document.getElementById("plan").innerHTML = s.plan.length ? s.plan.map(a=>ARROW[a]).join("") : "<span style='color:var(--dim);font-size:13px;letter-spacing:0'>explore</span>";
  const tracks=diverges();
  const st=document.getElementById("status");
  if(tracks===null){ st.innerHTML="<span class='status' style='color:var(--dim)'>—</span>"; }
  else if(tracks){ st.innerHTML="<span class='status' style='color:var(--good)'>&#10003; reality tracks it</span>"; }
  else { st.innerHTML="<span class='status' style='color:var(--warn)'>&#8776; diverges here</span>"; }
  renderReal(); renderImag();
}

function tickAnim(){
  const traj=grid().steps[T].imagined;
  if(subT < traj.length-1){ subT++; renderImag(); }
  else {
    if(T < grid().steps.length-1){ T++; subT=0; render(); }
    else { stop(); return; }
  }
}
function play(){ playing=true; document.getElementById("play").textContent="Pause";
  timer=setInterval(tickAnim, 420); }
function stop(){ playing=false; document.getElementById("play").textContent="Play"; clearInterval(timer); }
document.getElementById("play").onclick=()=>{ playing?stop():(T>=grid().steps.length-1&&(T=0),subT=0,render(),play()); };
scrub.oninput=e=>{ stop(); T=+e.target.value; subT=grid().steps[T].imagined.length-1; render(); };

window.addEventListener("resize",()=>{ resize(); render(); });
resize(); render();
</script>
"""

out = sys.argv[1] if len(sys.argv) > 1 else "imagination_widget.html"
with open(out, "w", encoding="utf-8") as f:
    f.write(HTML.replace("__DATA__", json.dumps(DATA)))
print("wrote", out, os.path.getsize(out), "bytes; won =", DATA["won"],
      "; levels =", [(lv["id"], len(lv["steps"])) for lv in DATA["levels"]])
