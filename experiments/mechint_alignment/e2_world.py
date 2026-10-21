"""E2: how fast does a world model learn each of Village's rules from refutations? (EXPERIMENT_E2.md)

Online: the training stream (transitions.py, time-major) is consumed once, in order, one time step of all 256 lanes
per update, like experience arriving. At log-spaced points the model is scored on held-out layouts, per rule:
- acc: the fraction of held-out instances whose check entries (the cells/fields the rule determines) are ALL
  predicted right (argmax);
- bits: -log2 p(true value) of those entries;
- seen: how many instances of the rule the stream had delivered so far (the refutation opportunities).
Overall: bits per entry, and accuracy on the entries that changed versus those that did not.

Learning rules (--rule):
- adam:   Adam on the summed per-field cross-entropy (the baseline).
- focal:  the same, each entry weighted by (1 - p_true)^gamma: attend to the refutations.
- rls:    the trunk by Adam; the readout heads by recursive least squares (a Kalman filter on the readout): the step
          in each feature direction is scaled by the inverse of how much that direction has been seen, so a refutation
          in a novel direction is corrected at once and familiar directions are protected.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from transitions import EVENTS, UNCHECKABLE
from worldmodel import WorldModel, flops_per_forward

LN2 = math.log(2)


def load(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def eval_set(test, per_event=256, n_random=2048, seed=0):
    rng = np.random.default_rng(seed)
    idx, by_event = [], {}
    for k, e in enumerate(EVENTS):
        if e in UNCHECKABLE:
            continue
        rows = np.nonzero((test["events"] >> k) & 1)[0]
        rows = rng.permutation(rows)[:per_event]
        by_event[e] = (len(idx), len(idx) + len(rows))
        idx.extend(rows.tolist())
    rand = rng.permutation(len(test["act"]))[:n_random]
    by_event["_random"] = (len(idx), len(idx) + len(rand))
    idx.extend(rand.tolist())
    idx = np.array(idx)
    sub = {k: test[k][idx] for k in ("obs", "gob", "nxt", "gnx", "act", "events", "checks")}
    return sub, by_event


def stack82(cell, glob):
    """[M,81,9] and [M,4] -> [M,82,9] with the global token as row 81 (fields 0-3)."""
    out = torch.zeros(cell.shape[0], 82, cell.shape[2], dtype=cell.dtype, device=cell.device)
    out[:, :81] = cell
    out[:, 81, :glob.shape[1]] = glob
    return out


@torch.no_grad()
def evaluate(model, ev, by_event, dev, chunk=1024):
    model.eval()
    M = len(ev["act"])
    pred = torch.zeros(M, 82, 9, dtype=torch.long)
    lp = torch.zeros(M, 82, 9)
    for i in range(0, M, chunk):
        sl = slice(i, i + chunk)
        o, g, a = (torch.from_numpy(ev[k][sl]).to(dev) for k in ("obs", "gob", "act"))
        tc, tg = torch.from_numpy(ev["nxt"][sl]).to(dev).long(), torch.from_numpy(ev["gnx"][sl]).to(dev).long()
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            cl, gl, _ = model(o, g, a)
        pc = torch.stack([x.argmax(-1) for x in cl], -1)
        pg = torch.stack([x.argmax(-1) for x in gl], -1)
        lc = torch.stack([x.float().log_softmax(-1).gather(-1, tc[..., f:f + 1]).squeeze(-1) for f, x in enumerate(cl)], -1)
        lg = torch.stack([x.float().log_softmax(-1).gather(-1, tg[..., f:f + 1]).squeeze(-1) for f, x in enumerate(gl)], -1)
        pred[sl] = stack82(pc, pg).cpu()
        lp[sl] = stack82(lc, lg).cpu()
    tgt = stack82(torch.from_numpy(ev["nxt"]).long(), torch.from_numpy(ev["gnx"]).long())
    cur = stack82(torch.from_numpy(ev["obs"]).long(), torch.from_numpy(ev["gob"]).long())
    out = {}
    for k, e in enumerate(EVENTS):
        if e not in by_event:
            continue
        lo, hi = by_event[e]
        if hi == lo:
            continue
        ch = torch.from_numpy(ev["checks"][lo:hi, k].astype(np.int64))      # [R, MAX_CHECKS, 2]
        valid = ch[..., 0] >= 0
        rows = torch.arange(lo, hi)[:, None].expand_as(ch[..., 0])
        r, f = ch[..., 0].clamp(min=0), ch[..., 1].clamp(min=0)
        ok = (pred[rows, r, f] == tgt[rows, r, f]) | ~valid
        bits = -(lp[rows, r, f] * valid).sum() / valid.sum() / LN2
        out[e] = dict(acc=float(ok.all(1).float().mean()), bits=float(bits), n=hi - lo)
    lo, hi = by_event["_random"]
    mask = torch.ones(82, 9, dtype=torch.bool)
    mask[81, 4:] = False
    t, p, c, l = tgt[lo:hi], pred[lo:hi], cur[lo:hi], lp[lo:hi]
    changed = (t != c) & mask
    same = (t == c) & mask
    per_field = -(l * mask).sum(dim=1).mean(0) / LN2                        # [9]: cell fields 0-8 (+ global in 0-3)
    out["_fields"] = {name: float(per_field[i]) for i, name in enumerate(
        ("terrain", "amount", "owner", "occupant", "occupant_id", "health", "trapped", "seen", "facing"))}
    out["_overall"] = dict(bits_per_transition=float(-(l * mask).sum(dim=(1, 2)).mean() / LN2),
                           changed_acc=float((p == t)[changed].float().mean()),
                           unchanged_acc=float((p == t)[same].float().mean()))
    model.train()
    return out


def robot_pos(obs):
    return (obs[..., 3] == 1).float().argmax(1)


def robot_mask(obs, radius):
    """[B, 81] bool: the cells within Chebyshev `radius` of the robot (occupant code 1) in obs [B, 81, 9]."""
    pos = (obs[..., 3] == 1).float().argmax(1)
    r, c = pos // 9, pos % 9
    rr = torch.arange(81, device=obs.device) // 9
    cc = torch.arange(81, device=obs.device) % 9
    return torch.maximum((rr[None] - r[:, None]).abs(), (cc[None] - c[:, None]).abs()) <= radius


def losses(cl, gl, nxt, gnx, gamma=0.0, per_row=False, mask=None, robot_pos=None):
    """The training loss: the sum over fields of the mean entry cross-entropy (focal-weighted if gamma). With per_row,
    also each row's total surprise in nats (unweighted, summed over all its entries)."""
    train = 0.0
    total = 0.0
    for lg, tgt in [(x, nxt[..., f]) for f, x in enumerate(cl)] + [(x, gnx[:, f]) for f, x in enumerate(gl)]:
        logits = lg.float() if lg.dim() == 2 else lg.float().permute(0, 2, 1)
        ce = F.cross_entropy(logits, tgt.long(), reduction="none")          # [B] or [B, 81]
        w = ce * (1 - torch.exp(-ce)).pow(gamma) if gamma else ce
        if mask is not None and w.dim() == 2:          # only the cells in the mask count
            w = w * mask * (mask.shape[1] / mask.sum(1, keepdim=True).clamp(min=1))
        train = train + (w if w.dim() == 1 else w.mean(1))
        total = total + (ce if ce.dim() == 1 else (ce if mask is None else ce * mask).sum(1))   # surprise in scope
    if per_row == "rows":                               # the training loss of each row, and its surprise
        return train, total.detach()
    if per_row == "sig":
        return train.mean(), total.detach(), signature(cl, gl, nxt, gnx, mask, robot_pos)
    return (train.mean(), total.detach()) if per_row else train.mean()


@torch.no_grad()
def signature(cl, gl, nxt, gnx, mask, robot_pos):
    """Each row's surprise KIND: the field of its most surprising in-scope entry and that entry's offset from the
    robot (clipped to +-2), or the global field: 9 * 25 + 4 = 229 kinds. Label-free: what changed, where relative to me."""
    cell_ce = torch.stack([F.cross_entropy(x.float().permute(0, 2, 1), nxt[..., f].long(), reduction="none")
                           for f, x in enumerate(cl)], 1)                     # [B, 9, 81]
    if mask is not None:
        cell_ce = cell_ce * mask[:, None]
    glob_ce = torch.stack([F.cross_entropy(x.float(), gnx[:, f].long(), reduction="none") for f, x in enumerate(gl)], 1)
    B = cell_ce.shape[0]
    best_cell, idx = cell_ce.view(B, -1).max(1)
    best_glob, gidx = glob_ce.max(1)
    f, c = idx // 81, idx % 81
    dr = (c // 9 - robot_pos // 9).clamp(-2, 2) + 2
    dc = (c % 9 - robot_pos % 9).clamp(-2, 2) + 2
    sig = f * 25 + dr * 5 + dc
    return torch.where(best_glob > best_cell, 225 + gidx, sig).cpu().numpy()


class Questions:
    """The open-questions buffer: rows the model could not predict, replayed until explained (loss below the median
    of the newest batch) or until replay stops helping (no drop of min_drop over the last `patience` replays)."""

    def __init__(self, cap, mix, patience, min_drop, admit):
        self.cap, self.mix, self.patience, self.min_drop, self.admit = cap, mix, patience, min_drop, admit
        self.rows, self.hist = [], []
        self.explained = self.stuck = self.admitted = 0

    def sample(self, rng):
        if not self.rows:
            return np.zeros(0, int), np.zeros(0, int)
        pos = rng.choice(len(self.rows), min(self.mix, len(self.rows)), replace=False)
        return pos, np.array([self.rows[i] for i in pos])

    def update(self, new_rows, new_loss, pos, rep_loss):
        median = float(np.median(new_loss))
        drop = set()
        for i, l in zip(pos, rep_loss):
            h = self.hist[i]
            h.append(float(l))
            if l < median:
                drop.add(i)
                self.explained += 1
            elif len(h) > self.patience and l > (1 - self.min_drop) * h[-1 - self.patience]:
                drop.add(i)
                self.stuck += 1
        keep = [i for i in range(len(self.rows)) if i not in drop]
        self.rows = [self.rows[i] for i in keep]
        self.hist = [self.hist[i] for i in keep]
        cut = np.quantile(new_loss, 1 - self.admit)
        for r, l in zip(new_rows, new_loss):
            if l > cut:
                self.rows.append(int(r))
                self.hist.append([float(l)])
                self.admitted += 1
        if len(self.rows) > self.cap:                     # over capacity: keep the least explained
            order = np.argsort([-h[-1] for h in self.hist])[:self.cap]
            self.rows = [self.rows[i] for i in order]
            self.hist = [self.hist[i] for i in order]


class RLSReadout:
    """Recursive least squares on the readout heads. S = decayed sum of feature outer products (+ a bias feature);
    step dW = -eta * (sum over entries of g h^T) (S + lam I)^-1, with g the logit gradient of each entry."""

    def __init__(self, model, eta, decay, lam, dev):
        d = model.d + 1
        self.model, self.eta, self.decay, self.lam = model, eta, decay, lam
        self.S = {"cell": torch.zeros(d, d, device=dev), "glob": torch.zeros(d, d, device=dev)}
        self.eye = torch.eye(d, device=dev)

    @torch.no_grad()
    def step(self, feats):
        hc, hg = feats
        for key, H, heads in (("cell", hc.reshape(-1, hc.shape[-1]), self.model.cell_heads),
                              ("glob", hg, self.model.glob_heads)):
            Ha = torch.cat([H.float(), torch.ones(H.shape[0], 1, device=H.device)], 1)
            self.S[key].mul_(self.decay).add_(Ha.T @ Ha)
            P = torch.linalg.inv(self.S[key] + self.lam * self.eye)
            n = H.shape[0]
            for m in heads:
                G = torch.cat([m.weight.grad, m.bias.grad[:, None]], 1) * n
                D = -self.eta * G @ P
                m.weight.add_(D[:, :-1])
                m.bias.add_(D[:, -1])
                m.weight.grad = None
                m.bias.grad = None


class Rumination:
    """Arm L. Open questions (rows the model could not predict) are rehearsed, the ones whose loss is falling first,
    until explained; then RETIRED to a long-term pool rehearsed rarely, and REOPENED if they become surprising again
    (forgotten). Open items whose loss stops falling are dropped (noise cannot be learned)."""

    def __init__(self, cap_open=4096, cap_ret=16384, mix_open=128, mix_ret=64, admit=0.10, patience=10,
                 min_drop=0.05, window=3):
        self.cap_open, self.cap_ret, self.mix_open, self.mix_ret = cap_open, cap_ret, mix_open, mix_ret
        self.admit, self.patience, self.min_drop, self.window = admit, patience, min_drop, window
        self.open_rows, self.open_hist, self.ret_rows = [], [], []
        self.admitted = self.retired = self.stuck = self.reopened = 0
        self.fair = False

    def _priority(self, h, rng):
        if len(h) <= self.window:
            return 1e9 + rng.random()                       # not yet rehearsed enough to judge: first
        return (h[-1 - self.window] - h[-1]) / max(h[-1 - self.window], 1e-6) + 1e-3 * rng.random()

    def sample(self, rng):
        op = np.zeros(0, int)
        if self.open_rows and self.fair:                    # L2: half by progress, half to the least rehearsed
            n = np.array([len(h) for h in self.open_hist])
            judged = np.nonzero(n > self.window)[0]
            pr = np.array([self._priority(self.open_hist[i], rng) for i in judged])
            top = judged[np.argsort(-pr)[:self.mix_open // 2]] if len(judged) else np.zeros(0, int)
            rest = np.setdiff1d(np.arange(len(n)), top)
            least = rest[np.argsort(n[rest] + 1e-3 * rng.random(len(rest)))[:self.mix_open - len(top)]]
            op = np.concatenate([top, least]).astype(int)
        elif self.open_rows:
            pr = np.array([self._priority(h, rng) for h in self.open_hist])
            op = np.argsort(-pr)[:self.mix_open]
        rp = rng.choice(len(self.ret_rows), min(self.mix_ret, len(self.ret_rows)), replace=False)             if self.ret_rows else np.zeros(0, int)
        return (op, np.array([self.open_rows[i] for i in op], int),
                rp, np.array([self.ret_rows[i] for i in rp], int))

    def update(self, new_rows, new_s, op, op_s, rp, rp_s):
        median, high = float(np.median(new_s)), float(np.quantile(new_s, 1 - self.admit))
        drop_open, drop_ret, to_open = set(), set(), []
        for i, sv in zip(op, op_s):
            h = self.open_hist[i]
            h.append(float(sv))
            if sv < median or (self.fair and sv < 0.5 * h[0]):
                drop_open.add(i)
                self.ret_rows.append(self.open_rows[i])
                self.retired += 1
            elif len(h) > self.patience and sv > (1 - self.min_drop) * h[-1 - self.patience]:
                drop_open.add(i)
                self.stuck += 1
        for i, sv in zip(rp, rp_s):
            if sv > high:                                   # forgotten: reopen
                drop_ret.add(i)
                to_open.append((self.ret_rows[i], float(sv)))
                self.reopened += 1
        keep = [i for i in range(len(self.open_rows)) if i not in drop_open]
        self.open_rows = [self.open_rows[i] for i in keep]
        self.open_hist = [self.open_hist[i] for i in keep]
        if drop_ret:
            self.ret_rows = [r for i, r in enumerate(self.ret_rows) if i not in drop_ret]
        for r, sv in to_open + [(int(r), float(sv)) for r, sv in zip(new_rows, new_s) if sv > high]:
            self.open_rows.append(int(r))
            self.open_hist.append([sv])
            self.admitted += 1
        if len(self.open_rows) > self.cap_open:             # keep the most surprising
            order = np.argsort([-h[-1] for h in self.open_hist])[:self.cap_open]
            self.open_rows = [self.open_rows[i] for i in order]
            self.open_hist = [self.open_hist[i] for i in order]
        if len(self.ret_rows) > self.cap_ret:               # forget the oldest retired
            self.ret_rows = self.ret_rows[-self.cap_ret:]


class KindRumination(Rumination):
    """Arm L3: rumination with rehearsal allocated across surprise KINDS (signature()) by their measured learning
    progress: each kind's EMA of the relative loss drop per rehearsal of its items. Untried kinds get an exploration
    bonus. Within a kind, the least-rehearsed items go first."""

    def __init__(self, n_kinds=229, ema=0.9, explore=0.05, **kw):
        super().__init__(**kw)
        self.fair = True
        self.open_sig = []
        self.prog = np.zeros(n_kinds)
        self.tries = np.zeros(n_kinds)
        self.ema, self.explore = ema, explore
        self.sig_of = {}

    def sample(self, rng):
        op = np.zeros(0, int)
        if self.open_rows:
            sig = np.array(self.open_sig)
            kinds = np.unique(sig)
            w = np.maximum(self.prog[kinds], 0) + np.where(self.tries[kinds] < 5, 1.0, self.explore)
            counts = rng.multinomial(self.mix_open, w / w.sum())
            n = np.array([len(h) for h in self.open_hist])
            picks = []
            for kd, cnt in zip(kinds, counts):
                if cnt:
                    items = np.nonzero(sig == kd)[0]
                    picks.append(items[np.argsort(n[items] + 1e-3 * rng.random(len(items)))[:cnt]])
            op = np.concatenate(picks).astype(int) if picks else op
        rp = rng.choice(len(self.ret_rows), min(self.mix_ret, len(self.ret_rows)), replace=False)             if self.ret_rows else np.zeros(0, int)
        return (op, np.array([self.open_rows[i] for i in op], int),
                rp, np.array([self.ret_rows[i] for i in rp], int))

    def update(self, new_rows, new_s, op, op_s, rp, rp_s, new_sig=None):
        for i, sv in zip(op, op_s):                         # learning progress per kind
            h, kd = self.open_hist[i], self.open_sig[i]
            drop = (h[-1] - float(sv)) / max(h[-1], 1e-6)
            self.prog[kd] = self.ema * self.prog[kd] + (1 - self.ema) * drop if self.tries[kd] else drop
            self.tries[kd] += 1
        self.sig_of.update({int(r): int(k) for r, k in zip(new_rows, new_sig)})
        super().update(new_rows, new_s, op, op_s, rp, rp_s)
        self.open_sig = [self.sig_of.get(r, 0) for r in self.open_rows]

    def top_kinds(self, n=6):
        tried = np.nonzero(self.tries >= 5)[0]
        order = tried[np.argsort(-self.prog[tried])][:n]
        return [(int(k), round(float(self.prog[k]), 3), int(self.tries[k])) for k in order]


def eval_points(steps):
    pts, x = {0}, 1.0
    while x < steps:
        pts.add(int(round(x)))
        x *= 1.3
    pts.add(steps)
    return sorted(pts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--rule", default="adam", choices=("adam", "focal", "rls"))
    p.add_argument("--pos", default="abs", choices=("abs", "rel", "local"))
    p.add_argument("--radius", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--d", type=int, default=192)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--gamma", type=float, default=2.0)
    p.add_argument("--eta", type=float, default=1.0)
    p.add_argument("--decay", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--persist", type=float, default=0.0, help="persistence prior strength (0 = off)")
    p.add_argument("--questions", type=int, default=0, help="open-questions buffer capacity (0 = off)")
    p.add_argument("--q-mix", type=int, default=128)
    p.add_argument("--q-patience", type=int, default=3)
    p.add_argument("--q-drop", type=float, default=0.1)
    p.add_argument("--q-admit", type=float, default=0.25)
    p.add_argument("--focus", type=int, default=-1, help="loss only within this radius of the robot (-1 = off)")
    p.add_argument("--ruminate", action="store_true", help="arm L: open questions -> retired explanations")
    p.add_argument("--fair", action="store_true", help="arm L2: progress/least-rehearsed split; retire when halved")
    p.add_argument("--kinds", action="store_true", help="arm L3: rehearsal allocated across surprise kinds by progress")
    p.add_argument("--replay", type=int, default=1, help="updates per arriving batch (each on a fresh mix)")
    p.add_argument("--steps", type=int, default=0, help="0 = the whole stream")
    p.add_argument("--batch", type=int, default=0, help="rows per update (0 = one time step of all lanes)")
    p.add_argument("--warmup", type=int, default=0, help="linear learning-rate warmup, in updates")
    p.add_argument("--time-limit", type=float, default=285.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data", default="runs/data")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda")
    torch.manual_seed(a.seed)
    train = load(os.path.join(a.data, "train.npz"))
    test = load(os.path.join(a.data, "test.npz"))
    L = a.batch or int(train["lanes"])                     # rows per update (the stream stays time-major)
    steps = a.steps or len(train["act"]) // L
    ev, by_event = eval_set(test)
    ev_counts = np.stack([((train["events"][:steps * L] >> k) & 1).reshape(-1, L).sum(1) for k in range(len(EVENTS))], 1).cumsum(0)

    model = WorldModel(a.d, a.layers, a.heads, pos=a.pos, persist=a.persist, radius=a.radius).to(dev)
    heads = set(id(q) for m in list(model.cell_heads) + list(model.glob_heads) for q in m.parameters())
    trunk = [q for q in model.parameters() if a.rule != "rls" or id(q) not in heads]
    opt = torch.optim.Adam(trunk, lr=a.lr)
    rls = RLSReadout(model, a.eta, a.decay, a.lam, dev) if a.rule == "rls" else None
    qs = Questions(a.questions, a.q_mix, a.q_patience, a.q_drop, a.q_admit) if a.questions else None
    rum = (KindRumination() if a.kinds else Rumination()) if a.ruminate else None
    if rum is not None:
        rum.fair = a.fair
    n_rows = 0
    fwd = flops_per_forward(model)
    json.dump(dict(vars(a), n_params=sum(q.numel() for q in model.parameters()), flops_per_forward=fwd,
                   steps=steps, lanes=L), open(os.path.join(a.out, "config.json"), "w"), indent=1)
    log = open(os.path.join(a.out, "log.jsonl"), "w")
    points = eval_points(steps)
    t_train, t0 = 0.0, time.time()
    rng = np.random.default_rng(a.seed)
    for s in range(steps + 1):
        if s in points:
            r = evaluate(model, ev, by_event, dev)
            row = dict(step=s, transitions=s * L, train_seconds=t_train, flops=3 * fwd * (n_rows or s * L * a.replay))
            if qs is not None:
                row["questions"] = dict(size=len(qs.rows), admitted=qs.admitted, explained=qs.explained, stuck=qs.stuck)
            if rum is not None:
                row["rumination"] = dict(open=len(rum.open_rows), retired_pool=len(rum.ret_rows), admitted=rum.admitted,
                                         retired=rum.retired, stuck=rum.stuck, reopened=rum.reopened)
                if a.kinds:
                    row["rumination"]["top_kinds"] = rum.top_kinds()
            for k, e in enumerate(EVENTS):
                if e in r:
                    r[e]["seen"] = int(ev_counts[s - 1, k]) if s > 0 else 0
            row.update(r)
            log.write(json.dumps(row) + "\n")
            log.flush()
            o = r["_overall"]
            print(f"step {s:5d}  {t_train:6.1f}s  bits/transition {o['bits_per_transition']:8.2f}  changed acc "
                  f"{o['changed_acc']:.3f}  theft {r['theft']['acc']:.2f}  mined {r['mined']['acc']:.2f}  "
                  f"push {r['push']['acc']:.2f}  button {r['button_off']['acc']:.2f}", flush=True)
        if s == steps or t_train > a.time_limit:
            break
        t1 = time.time()
        if a.warmup:
            for gr in opt.param_groups:
                gr["lr"] = a.lr * min(1.0, (s + 1) / a.warmup)
        for rep in range(a.replay):
            if rep == 0:
                rows = np.arange(s * L, (s + 1) * L)
            else:                                             # replay: a random batch from what has arrived
                rows = rng.integers(0, (s + 1) * L, L)
            pos = np.zeros(0, int)
            if qs is not None and rep == 0:
                pos, qrows = qs.sample(rng)
                rows = np.concatenate([rows, qrows])
            if rum is not None and rep == 0:
                op, orows, rp, rrows = rum.sample(rng)
                rows = np.concatenate([rows, orows, rrows])
            n_rows += len(rows)
            o, g, act, nx, gx = (torch.from_numpy(train[k][rows]).to(dev, non_blocking=True)
                                 for k in ("obs", "gob", "act", "nxt", "gnx"))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                cl, gl, feats = model(o, g, act)
            out = losses(cl, gl, nx, gx, a.gamma if a.rule == "focal" else 0.0, per_row="sig" if a.kinds else True,
                         mask=robot_mask(o, a.focus) if a.focus >= 0 else None, robot_pos=robot_pos(o))
            loss, surprise = out[0], out[1]
            if qs is not None and rep == 0:
                sp = surprise.cpu().numpy()
                qs.update(rows[:L], sp[:L], pos, sp[L:])
            if rum is not None and rep == 0:
                sp = surprise.cpu().numpy()
                if a.kinds:
                    rum.update(rows[:L], sp[:L], op, sp[L:L + len(op)], rp, sp[L + len(op):], new_sig=out[2][:L])
                else:
                    rum.update(rows[:L], sp[:L], op, sp[L:L + len(op)], rp, sp[L + len(op):])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if rls is not None:
                rls.step((feats[0].detach(), feats[1].detach()))
            opt.step()
        torch.cuda.synchronize()
        t_train += time.time() - t1
    torch.save(model.state_dict(), os.path.join(a.out, "model.pt"))
    print(f"done in {time.time() - t0:.0f}s wall ({t_train:.0f}s training)")


if __name__ == "__main__":
    main()
