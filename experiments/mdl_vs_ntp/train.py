"""The main loop for every arm. One entrypoint, one budget, the arm decides what runs between optimizer steps.

    python experiments/mdl_vs_ntp/train.py --arm mdl --seed 0
    python experiments/mdl_vs_ntp/train.py --arm ntp --seed 0 --measure_throughput 200     (sets B, Phase 0)

Compute is accounted in TOKEN-EQUIVALENTS: 3× non-PAD tokens per training or REINFORCE step, 1× tokens per forward-only
pass (probes, wake and proposer sampling as prompt + generated). CPU work (execution, DP scoring) is logged as seconds,
not counted. Every run stops at the same budget B, so `ntp` takes more optimizer steps than the self-training arms.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from env.families import load_env, make_instance                                        # noqa: E402
from env.grammar import Grammar                                                          # noqa: E402
from env.tokens import (LEN_I, PAD, PROG_BUDGET, encode_I, encode_T, prim_units, prog_tokens,  # noqa: E402
                        probe_positions)
from mdl.curriculum import Curriculum                                                    # noqa: E402
from mdl.library import Library, segment_units                                           # noqa: E402
from mdl.proposer import propose_round                                                   # noqa: E402
from mdl.wake import Buffer, buffer_batch, wake_round                                    # noqa: E402
from model_adapter import ModelAdapter, causality_test                                   # noqa: E402

ARMS = {
    #            sampling   wake   target        library  scoring
    "ntp":          ("uniform", False, "gt",       False, None),
    "exit":         ("uniform", True,  "random",   False, None),
    "mdl":          ("gain",    True,  "mdl",      True,  "disjoint"),
    "mdl_nolib":    ("gain",    True,  "mdl",      False, None),
    "curio":        ("curio",   True,  "mdl",      True,  "disjoint"),
    "mdl_insample": ("gain",    True,  "mdl",      True,  "insample"),
}
BATCH, N_BUFFER = 64, 16
PROBE_POS = probe_positions()


def load_config():
    with open(HERE / "config.yaml") as fh:
        return yaml.safe_load(fh)


class Run:
    def __init__(self, args, cfg):
        self.args, self.cfg = args, cfg
        self.arm = args.arm
        self.sampling, self.do_wake, self.target, self.has_lib, self.scoring = ARMS[self.arm]
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.out = Path(args.out) if args.out else HERE / "runs" / f"{self.arm}_s{args.seed}"
        self.out.mkdir(parents=True, exist_ok=True)
        self.env = load_env()
        self.train_fids = self.env.train_fids
        self.categories = {f: self.env.fam(f).category for f in self.train_fids}

        torch.manual_seed(args.seed)
        random.seed(args.seed)
        self.rng = np.random.default_rng(args.seed)
        self.trng = torch.Generator(device=self.dev).manual_seed(args.seed)

        self.adapter = ModelAdapter(args.size, amp=cfg.get("amp", True)).to(self.dev)
        self.adapter.maybe_compile(cfg.get("compile", False))
        self.opt = torch.optim.AdamW(self.adapter.parameters(), lr=args.lr, betas=(0.9, 0.95),
                                     weight_decay=cfg.get("weight_decay", 0.01))
        self.B = int(args.budget * args.budget_frac)
        self.round_te = int(args.round_pct * self.B)
        self.ckpt_te = [int(self.B * q) for q in (0.25, 0.5, 0.75, 1.0)]
        self.library = Library()
        self.grammar = Grammar(self.library if self.has_lib else None)
        self.buffer = Buffer(per_entry=1 if self.arm == "exit" else 4)
        self.curriculum = Curriculum(self.train_fids, mode={"uniform": "uniform", "gain": "gain",
                                                            "curio": "curio"}[self.sampling])
        self.te = self.step = 0
        self.next_round = self.round_te
        self.cpu_seconds = 0.0
        self.t_start = time.time()
        self.logs = {k: open(self.out / f"{k}.jsonl", "a") for k in ("train", "rounds", "wake", "probe")}
        with open(self.out / "config.yaml", "w") as fh:
            yaml.safe_dump({**cfg, **vars(args), "B_effective": self.B, "params": self.adapter.num_params()}, fh)

    # ── schedule ─────────────────────────────────────────────────────────────────────────────────────────────────────
    def lr_at(self, frac):
        warm = 0.02
        if frac < warm:
            return self.args.lr * frac / warm
        t = min(1.0, (frac - warm) / (1 - warm))
        return self.args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))

    def set_lr(self):
        lr = self.lr_at(self.te / self.B)
        for g in self.opt.param_groups:
            g["lr"] = lr
        return lr

    # ── data ─────────────────────────────────────────────────────────────────────────────────────────────────────────
    def sample_fids(self, n):
        if self.sampling == "uniform":
            return [self.train_fids[i] for i in self.rng.integers(len(self.train_fids), size=n)]
        return self.curriculum.sample(self.rng, n)

    def supervised(self, n):
        seqs, masks, kinds = [], [], []
        for fid in self.sample_fids(n):
            fam = self.env.fam(fid)
            inst = make_instance(fam, int(self.rng.integers(1 << 30)), "train")
            if fam.kind == "structured" and fam.labeled and self.rng.random() < 0.5:
                prog = fam.program(inst.args)
                units = segment_units(prog, self.library) if self.has_lib else prim_units(prog)
                toks = prog_tokens(units)
                if len(toks) + 1 > PROG_BUDGET:
                    toks = prog_tokens(prim_units(prog))
                s, m = encode_I(inst, toks)
                kinds.append(("I", fam.category))
            else:
                s, m = encode_T(inst)
                kinds.append(("T", fam.category))
            seqs.append(s)
            masks.append(m)
        return seqs, masks, kinds

    def make_batch(self):
        n_buf = 0 if (self.arm == "ntp" or len(self.buffer) == 0) else N_BUFFER
        seqs, masks, kinds = self.supervised(BATCH - n_buf)
        if n_buf:
            t0 = time.time()
            extra = buffer_batch(self.buffer, self.env, self.library, self.arm, n_buf, self.rng)
            self.cpu_seconds += time.time() - t0
            for s, m in extra:
                seqs.append(s)
                masks.append(m)
                kinds.append(("B", "buffer"))
            if len(extra) < n_buf:
                s2, m2, k2 = self.supervised(n_buf - len(extra))
                seqs += s2
                masks += m2
                kinds += k2
        L = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), L), PAD, dtype=torch.long)
        msk = torch.zeros((len(seqs), L), dtype=torch.bool)
        for i, (s, m) in enumerate(zip(seqs, masks)):
            ids[i, :len(s)] = torch.tensor(s)
            msk[i, :len(m)] = torch.tensor(m)
        return ids.to(self.dev), msk.to(self.dev), kinds

    # ── rounds ───────────────────────────────────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def probe_all(self):
        """Probe NLL (bits/token over demos 2-4 outputs + answer) for every train family; forward-only TE."""
        self.adapter.eval()
        seqs, owners = [], []
        for fid in self.train_fids:
            for inst in self.env.probe[fid]:
                seqs.append(encode_T(inst)[0])
                owners.append(fid)
        ids = torch.tensor(seqs, dtype=torch.long)
        per = collections.defaultdict(list)
        for s in range(0, len(ids), 256):
            chunk = ids[s:s + 256].to(self.dev)
            bits, _ = self.adapter.probe_bits(chunk, PROBE_POS)
            for fid, b in zip(owners[s:s + 256], bits.tolist()):
                per[fid].append(b)
        self.te += int(ids.numel())
        self.adapter.train()
        return {fid: float(np.mean(v)) for fid, v in per.items()}

    def do_round(self, rnd):
        rec = dict(round=rnd, te=self.te, step=self.step)
        if self.sampling != "uniform":
            losses = self.probe_all()
            p = self.curriculum.update(losses)
            noisy_loss = np.mean([losses[f] for f in self.env.by_split["noisy"]])
            rec.update(probe_mean=float(np.mean(list(losses.values()))), noisy_probe=float(noisy_loss),
                       mass=self.curriculum.mass_by(self.categories))
            self.logs["probe"].write(json.dumps(rec) + "\n")
        t0 = time.time()
        stats, te = wake_round(self.adapter, self.env, self.library, self.grammar, self.sample_fids, self.rng,
                               self.trng, self.buffer, rnd, self.dev)
        self.cpu_seconds += time.time() - t0
        self.te += te
        stats.update(round=rnd, te=self.te, buffer=len(self.buffer))
        self.logs["wake"].write(json.dumps(stats) + "\n")
        if self.has_lib:
            used = {t - 41 for t in stats["macro_usage"]}          # macro tokens -> slots (M0 = 41)
            t0 = time.time()
            rrec, te = propose_round(self.adapter, self.opt, self.env, self.library, self.buffer, self.grammar,
                                     self.arm, rnd, self.rng, self.trng, self.dev, self.cfg.get("beta", 0.1), used)
            self.cpu_seconds += time.time() - t0
            self.te += te
            rrec.update(te=self.te)
            self.logs["rounds"].write(json.dumps(rrec) + "\n")
            self.grammar = Grammar(self.library)
        for fh in self.logs.values():
            fh.flush()

    # ── checkpoints ──────────────────────────────────────────────────────────────────────────────────────────────────
    def save(self, tag):
        st = dict(model=self.adapter.state_dict(), opt=self.opt.state_dict(), te=self.te, step=self.step,
                  next_round=self.next_round, library=self.library.state(), curriculum=self.curriculum.state(),
                  buffer=self.buffer.state(), rng=self.rng.bit_generator.state, trng=self.trng.get_state(),
                  torch_rng=torch.get_rng_state(), cpu_seconds=self.cpu_seconds, ckpt_te=self.ckpt_te,
                  wall=time.time() - self.t_start)
        torch.save(st, self.out / f"ckpt_{tag}.pt")
        torch.save(st, self.out / "ckpt_latest.pt")

    def load(self, path):
        st = torch.load(path, map_location=self.dev, weights_only=False)
        self.adapter.load_state_dict(st["model"])
        self.opt.load_state_dict(st["opt"])
        self.te, self.step, self.next_round = st["te"], st["step"], st["next_round"]
        self.library.load_state(st["library"])
        self.grammar = Grammar(self.library if self.has_lib else None)
        self.curriculum.load_state(st["curriculum"])
        self.buffer.load_state(st["buffer"])
        self.rng.bit_generator.state = st["rng"]
        self.trng.set_state(st["trng"])
        torch.set_rng_state(st["torch_rng"])
        self.cpu_seconds, self.ckpt_te = st["cpu_seconds"], st["ckpt_te"]
        self.t_start = time.time() - st["wall"]

    # ── the loop ─────────────────────────────────────────────────────────────────────────────────────────────────────
    def train(self):
        if self.step == 0 and self.sampling != "uniform":
            losses = self.probe_all()                                        # baseline at TE = 0
            self.curriculum.update(losses)
        rnd = 0
        last_log, tok_window = time.time(), 0
        while self.te < self.B:
            if self.do_wake and self.te >= self.next_round:
                rnd = self.next_round // self.round_te
                self.do_round(rnd)
                self.next_round += self.round_te
            ids, msk, kinds = self.make_batch()
            lr = self.set_lr()
            self.adapter.train()
            logits, aux = self.adapter(ids)
            tgt = ids[:, 1:]
            m = msk[:, 1:]
            per_tok = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), tgt.reshape(-1),
                                      reduction="none").reshape(tgt.shape)
            loss = (per_tok * m).sum() / m.sum()
            if aux is not None:
                loss = loss + aux
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {self.step}")
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(self.adapter.parameters(), 1.0)
            self.opt.step()
            ntok = int((ids != PAD).sum())
            self.te += 3 * ntok
            tok_window += ntok
            self.step += 1
            if self.step % 50 == 0:
                by = {}
                for k in ("T", "I", "B"):
                    sel = torch.tensor([kk[0] == k for kk in kinds], device=self.dev)
                    if sel.any():
                        mm = m[sel]
                        by[k] = float((per_tok[sel] * mm).sum() / mm.sum())
                comp = dict(collections.Counter(kk[1] for kk in kinds))
                now = time.time()
                self.logs["train"].write(json.dumps(dict(
                    step=self.step, te=self.te, lr=lr, loss=float(loss), loss_by=by, grad_norm=float(gn),
                    tok_s=tok_window / max(now - last_log, 1e-6), wall=now - self.t_start,
                    batch=comp, cpu_seconds=self.cpu_seconds)) + "\n")
                self.logs["train"].flush()
                last_log, tok_window = now, 0
            while self.ckpt_te and self.te >= self.ckpt_te[0]:
                q = self.ckpt_te.pop(0)
                self.save(f"{int(round(100 * q / self.B))}")
        self.save("final")
        for fh in self.logs.values():
            fh.close()


def measure_throughput(run, n_steps):
    """ntp steps for `n_steps`, reporting tokens/s and the B that makes one ntp run ~30 minutes."""
    t0, te0 = time.time(), run.te
    for _ in range(n_steps):
        ids, msk, _k = run.make_batch()
        logits, _ = run.adapter(ids)
        tgt, m = ids[:, 1:], msk[:, 1:]
        loss = (F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), tgt.reshape(-1),
                                reduction="none").reshape(tgt.shape) * m).sum() / m.sum()
        run.opt.zero_grad(set_to_none=True)
        loss.backward()
        run.opt.step()
        run.te += 3 * int((ids != PAD).sum())
    dt = time.time() - t0
    te_s = (run.te - te0) / dt
    print(f"{n_steps} ntp steps in {dt:.1f}s | {te_s:.0f} TE/s | B for 30 min = {int(te_s * 1800)}")
    return int(te_s * 1800)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", default="4M")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--budget", type=int, default=None, help="B in token-equivalents (default: config.yaml)")
    ap.add_argument("--budget_frac", type=float, default=1.0, help="fraction of B to run (smoke: 0.02, pilot: 1/3)")
    ap.add_argument("--round_pct", type=float, default=0.05, help="round every this fraction of B (smoke: 0.005)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--measure_throughput", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config()
    if args.lr is None:
        args.lr = float(cfg.get("lr", 1e-3))
    if args.budget is None:
        if cfg.get("B") is None and not args.measure_throughput:
            raise SystemExit("B is not set: run `--arm ntp --measure_throughput 200` and record B in config.yaml")
        args.budget = int(cfg.get("B") or 10 ** 9)
    run = Run(args, cfg)
    worst = causality_test(run.adapter)
    assert worst < 1e-5, f"causality test failed: {worst}"
    print(f"arm {args.arm} seed {args.seed} | params {run.adapter.num_params():,} | B {run.B:,} | dev {run.dev} "
          f"| causality max diff {worst:.2e}")
    if args.measure_throughput:
        measure_throughput(run, args.measure_throughput)
        return
    if args.resume:
        run.load(args.resume)
    run.train()


if __name__ == "__main__":
    main()
