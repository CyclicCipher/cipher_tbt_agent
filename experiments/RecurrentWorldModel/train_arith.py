"""Few-shot arithmetic harness (Docs/FEW_SHOT_ARITHMETIC.md).

Trains the base PoPE transformer on ModularArithmetic from N labeled pairs and measures held-out
accuracy vs N — the sample-complexity curve. Arms (objectives):
  * A0 "ce"  -- plain CE (control; predicted to memorize, held-out -> chance).
  * A1 "tbt" -- CE + reference-frame coherence (cycle-consistency) + SIGReg  [added next increment].

Value-as-coord: 2-axis PoPE, axis0 = sequence position, axis1 = numeral value (the number line).
Tiny + few examples + CPU => sub-second runs; iterate freely. NOT a long training loop (Mistake #36).
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import copy  # noqa: E402

from baselines import FixedDepthConfig, FixedDepthTransformer  # noqa: E402
from baselines.sigreg import SIGReg  # noqa: E402
from tasks.arithmetic import ModularArithmetic  # noqa: E402
from tasks.number_line import NumberLineDiscovery  # noqa: E402
from tasks.vocab import VAL0  # noqa: E402
from train_numberline import train_discovery  # noqa: E402

EQ_POS = 3   # the scored position in [VAL(a), OP, VAL(b), EQ, VAL(c)]


def build_model(task, dim=64, n_heads=4, n_layers=2, seed=0) -> FixedDepthTransformer:
    torch.manual_seed(seed)
    cfg = FixedDepthConfig(
        vocab_size=task.vocab_size, dim=dim, n_heads=n_heads, n_layers=n_layers,
        max_seq=task.seq_len + 1, pos_mode="pope", n_axes=2,
    )
    return FixedDepthTransformer(cfg)


def ce_at_eq(logits, batch) -> torch.Tensor:
    v = logits.shape[-1]
    ce = F.cross_entropy(logits.reshape(-1, v), batch.targets.reshape(-1), reduction="none")
    ce = ce.reshape(batch.targets.shape)
    return (ce * batch.loss_mask).sum() / batch.loss_mask.sum().clamp_min(1.0)


@torch.no_grad()
def accuracy(model, task, pairs, device, max_eval=512) -> float:
    if not pairs:
        return float("nan")
    b = task.batch(pairs[:max_eval]).to(device)
    pred = model(b.input_ids, coord=b.coord)[:, EQ_POS].argmax(-1)
    return (pred == b.targets[:, EQ_POS]).float().mean().item()


def train_ce(task, n_train, steps=400, lr=3e-3, weight_decay=0.01, dim=64, seed=0, device="cpu"):
    model = build_model(task, dim=dim, seed=seed).to(device)
    train, held = task.split(n_train, seed=seed)
    b = task.batch(train).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    for _ in range(steps):
        loss = ce_at_eq(model(b.input_ids, coord=b.coord), b)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()
    return accuracy(model, task, train, device), accuracy(model, task, held, device)


# --- Phase 1: few-shot addition GIVEN a discovered number-line frame (BLUEPRINT §D step 3) ---

def discovered_model(m=17, steps=800, w_equiv=1.0, dim=64, seed=0, device="cpu"):
    """Phase 0: discover the number-line frame (succ/pred/compare + rotation equivariance); return the
    trained transformer to use as the phase-1 init. The arithmetic and discovery models are
    architecturally identical (shared vocab, dim, max_seq), so this is a drop-in initialisation."""
    model, _ = train_discovery(NumberLineDiscovery(m, seed=seed), steps=steps, w_equiv=w_equiv,
                               dim=dim, seed=seed, device=device)
    return model


def train_ce_from(model, task, n_train, steps=400, lr=1e-3, weight_decay=0.01, seed=0, device="cpu"):
    """Few-shot addition from a given init (e.g. the discovered frame): CE on N pairs, held-out acc.
    Lower lr than the random-init A0 so the discovered geometry isn't immediately overwritten."""
    train, held = task.split(n_train, seed=seed)
    b = task.batch(train).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    for _ in range(steps):
        loss = ce_at_eq(model(b.input_ids, coord=b.coord), b)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()
    return accuracy(model, task, train, device), accuracy(model, task, held, device)


def phase1_sweep(m=17, Ns=(2, 3, 4, 6, 8, 10), steps_disc=800, steps_add=400, w_equiv=1.0,
                 lr_add=1e-3, dim=64, seed=0, device="cpu"):
    """Headline test: does the discovered number line make addition few-shot, where random-init CE
    (A0) sits at chance? Discover the frame ONCE, then few-shot addition from a fresh copy per N."""
    task = ModularArithmetic(m, "add", seed=seed)        # value_coord=False (honest: learned frame)
    print(f"[phase-1] add m={m}  chance={1.0/m:.3f}")
    print("  A0  random init + CE:")
    for n in Ns:
        tr, ho = train_ce(task, n, steps=steps_add, lr=lr_add, dim=dim, seed=seed, device=device)
        print(f"    N={n:>2}  train {tr:.2f}  held {ho:.3f}", flush=True)
    base = discovered_model(m, steps=steps_disc, w_equiv=w_equiv, dim=dim, seed=seed, device=device)
    print("  (b) discovered frame + CE:")
    for n in Ns:
        tr, ho = train_ce_from(copy.deepcopy(base), task, n, steps=steps_add, lr=lr_add,
                               seed=seed, device=device)
        print(f"    N={n:>2}  train {tr:.2f}  held {ho:.3f}", flush=True)
    print("  (b-coh) discovered frame + coherence (A1):")
    for n in Ns:
        tr, ho = train_tbt(task, n, steps=steps_add, lr=lr_add, seed=seed, device=device,
                           init_model=copy.deepcopy(base))
        print(f"    N={n:>2}  train {tr:.2f}  held {ho:.3f}", flush=True)
    print("  (coh-rand) random init + coherence (A1, control):")
    for n in Ns:
        tr, ho = train_tbt(task, n, steps=steps_add, lr=lr_add, seed=seed, device=device)
        print(f"    N={n:>2}  train {tr:.2f}  held {ho:.3f}", flush=True)


# --- A1: the Thousand-Brains objective (prediction + reference-frame coherence + SIGReg) ---

def _logits_from_embeds(model, embeds, coord):
    """Replicate FixedDepthTransformer.forward but from pre-computed input embeddings (so the first
    operand can be a *soft* token for the coherence chain). pos is None under PoPE."""
    h = embeds
    if model.pos is not None:
        h = h + model.pos[:, : embeds.shape[1]]
    zero = torch.zeros_like(h)
    for i, layer in enumerate(model.layers):
        h = layer(h, zero, coord=coord)
        if model.inject is not None and i == model.inject_layer:
            h = h + model.inject(h)
    h = model.norm_out(h)
    return F.linear(h, model.embed.weight) if model.head_proj is None else model.head_proj(h)


def _vlogits_discrete(model, task, a, b, device):
    """value-logits (K, m) of f(a,b): the model's answer distribution over values at the EQ slot."""
    pairs = list(zip(a.tolist(), b.tolist()))
    batch = task.batch(pairs).to(device)
    logits = model(batch.input_ids, coord=batch.coord)
    return logits[:, EQ_POS, VAL0:VAL0 + task.P], batch


def _vlogits_soft_a(model, task, p_a, b, device):
    """value-logits (K, m) of f(soft_a, b), where the first operand is a distribution p_a over values
    (the coherence chain feeds f(a,b1)'s soft output back as the next location)."""
    m = task.P
    pairs = [(0, bb) for bb in b.tolist()]                 # placeholder a=0; we override slot 0
    batch = task.batch(pairs).to(device)
    embeds = model.embed(batch.input_ids) * model.cfg.emb_scale
    value_emb = model.embed.weight[VAL0:VAL0 + m]          # (m, dim)
    embeds = embeds.clone()
    embeds[:, 0, :] = (p_a @ value_emb) * model.cfg.emb_scale
    coord = batch.coord.clone()
    arange_m = torch.arange(m, dtype=torch.float, device=device)
    coord[:, 0, 1] = p_a @ arange_m                        # expected value on the number-line axis
    logits = _logits_from_embeds(model, embeds, coord)
    return logits[:, EQ_POS, VAL0:VAL0 + m]


def train_tbt(task, n_train, steps=400, lr=3e-3, weight_decay=0.01, dim=64, seed=0, device="cpu",
              k_triples=64, w_coh=1.0, w_id=1.0, w_sig=0.1, log_every=0, init_model=None):
    torch.manual_seed(seed)
    # init_model lets phase-1 start coherence from the *discovered* frame (vs a random one, which failed)
    model = (init_model if init_model is not None else build_model(task, dim=dim, seed=seed)).to(device)
    train, held = task.split(n_train, seed=seed)
    lab = task.batch(train).to(device)
    sig = SIGReg(n_slices=256).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    m = task.P
    g = torch.Generator(device=device).manual_seed(seed + 1)
    e_id = 0 if task.op == "add" else 1                    # the no-movement / identity displacement
    model.train()
    for step in range(steps):
        # (1) prediction on the labeled anchors
        pred_loss = ce_at_eq(model(lab.input_ids, coord=lab.coord), lab)
        # sampled unlabeled triples (a, b1, b2)
        a = torch.randint(0, m, (k_triples,), generator=g)
        b1 = torch.randint(0, m, (k_triples,), generator=g)
        b2 = torch.randint(0, m, (k_triples,), generator=g)
        comp = (a.new_tensor([task.apply(int(x), int(y)) for x, y in zip(b1, b2)]))  # b1 ⊕ b2
        # (2) coherence: f(f(a,b1),b2) == f(a, b1⊕b2)
        vl1, b1batch = _vlogits_discrete(model, task, a, b1, device)
        p1 = vl1.softmax(-1)
        p_chain = _vlogits_soft_a(model, task, p1, b2, device).softmax(-1)
        p_direct = _vlogits_discrete(model, task, a, comp, device)[0].softmax(-1)
        coh_loss = ((p_chain - p_direct) ** 2).sum(-1).mean()
        # (3) identity: f(a, e_id) == a
        vli = _vlogits_discrete(model, task, a, torch.full_like(a, e_id), device)[0]
        id_loss = F.cross_entropy(vli, a.to(device))
        # (4) SIGReg anti-collapse on the sampled hiddens (EQ position)
        hid = model.encode(b1batch.input_ids, b1batch.coord)[:, EQ_POS]   # (K, dim)
        sig_loss = sig(hid, generator=g)
        loss = pred_loss + w_coh * coh_loss + w_id * id_loss + w_sig * sig_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if log_every and (step % log_every == 0 or step == steps - 1):
            model.eval()
            ho = accuracy(model, task, held, device)
            model.train()
            print(f"    step {step:>4}  pred {pred_loss.item():.3f}  coh {coh_loss.item():.4f}  "
                  f"id {id_loss.item():.3f}  sig {sig_loss.item():.3f}  heldout {ho:.3f}", flush=True)
    model.eval()
    return accuracy(model, task, train, device), accuracy(model, task, held, device)


TRAINERS = {"ce": train_ce, "tbt": train_tbt}


def sweep(arm="ce", op="add", Ns=(2, 3, 4, 6, 8, 10), modulus=17, steps=400, dim=64, seed=0):
    task = ModularArithmetic(modulus, op, seed=seed)
    print(f"[{arm.upper()}] op={op} m={modulus} chance={1.0/modulus:.3f}")
    train = TRAINERS[arm]
    for n in Ns:
        tr, ho = train(task, n, steps=steps, dim=dim, seed=seed)
        print(f"  N={n:>2}  train_acc={tr:.2f}  heldout_acc={ho:.3f}", flush=True)


if __name__ == "__main__":
    sweep(arm="ce", op="add")
    sweep(arm="tbt", op="add")
