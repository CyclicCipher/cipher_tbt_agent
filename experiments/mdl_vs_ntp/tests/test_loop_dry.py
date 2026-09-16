"""Every code path of a round, exercised on CPU with a 1M model and NO optimizer step: batch assembly for each arm,
probes + curriculum update, wake, buffer replay, scoring/mining, the REINFORCE loss, and checkpoint round-trip."""
import argparse
import math

import numpy as np
import torch

from env.dsl import parse_pattern
from env.grammar import Grammar
from env.tokens import EDEF, LEN_P, PAD, PROMPT_P, encode_P, parse_def, prog_tokens
from mdl.library import Library, Scorer, segment_units
from mdl.proposer import build_archive, candidate_reward, mine
from mdl.wake import Buffer, buffer_batch, wake_round
from train import Run, load_config


def _run(tmp_path, arm, env):
    cfg = load_config()
    cfg["amp"] = False
    cfg["device"] = "cpu"                                  # keep the dry run off the GPU
    args = argparse.Namespace(arm=arm, seed=0, size="1M", lr=1e-3, budget=100000, budget_frac=1.0,
                              round_pct=0.05, out=str(tmp_path / arm), resume=None, measure_throughput=0)
    r = Run(args, cfg)
    r.env = env
    return r


def test_batches_probes_and_checkpoint_for_every_arm(tmp_path, env):
    for arm in ("ntp", "exit", "mdl", "mdl_nolib", "curio", "mdl_insample"):
        r = _run(tmp_path, arm, env)
        if r.has_lib:
            r.library.add(parse_pattern("sort rev"), 0)
            r.grammar = Grammar(r.library)
        ids, msk, kinds = r.make_batch()
        assert ids.shape[0] == 64 and msk.shape == ids.shape and msk.any()
        logits, _ = r.adapter(ids)
        loss = torch.nn.functional.cross_entropy(logits[:, :-1][msk[:, 1:]], ids[:, 1:][msk[:, 1:]])
        assert torch.isfinite(loss)
        if r.sampling != "uniform":
            losses = r.probe_all()
            assert len(losses) == 540 and all(math.isfinite(v) for v in losses.values())
            p = r.curriculum.update(losses)
            assert abs(p.sum() - 1) < 1e-6 and p.max() <= 0.05 + 0.1 / 540 + 1e-6
        r.te = 12345
        r.save("dry")
        r2 = _run(tmp_path, arm, env)
        r2.load(tmp_path / arm / "ckpt_dry.pt")
        assert r2.te == 12345 and len(r2.library) == len(r.library)
        for fh in r.logs.values():
            fh.close()
        for fh in r2.logs.values():
            fh.close()


def test_wake_buffer_scoring_and_reinforce_loss(tmp_path, env):
    r = _run(tmp_path, "mdl", env)
    r.library.add(parse_pattern("fgt ? sort"), 0)
    r.grammar = Grammar(r.library)
    stats, te = wake_round(r.adapter, r.env, r.library, r.grammar, r.sample_fids, r.rng, r.trng, r.buffer, 1,
                           r.dev, W=4, N=2)
    assert stats["tasks"] == 4 and te > 0
    # seed the buffer with a known-correct program so replay has something to verify
    fid = next(f for f in env.by_split["train"] if env.fam(f).labeled and env.fam(f).n_units >= 2)
    inst, prog = env.archive[fid][0]
    r.buffer.add(fid, inst.args, [prog], r.rng)
    seqs = buffer_batch(r.buffer, r.env, r.library, "mdl", 4, r.rng)
    assert seqs and all(len(s) == 108 for s, _m in seqs)
    # scoring + mining on the real archive
    arch = build_archive(env, r.buffer)
    S = list(arch)[:120]
    sc = Scorer({f: arch[f] for f in S}, r.library, family_avg=True)
    best, pat = mine(arch, S, sc, top=30)
    assert math.isfinite(best) and pat
    # the REINFORCE pieces, without the step: generate DEFs, reward them, and take the masked sequence log-prob
    ctx = [arch[f][0] for f in S[:5]]
    prompts = torch.tensor([encode_P([prog_tokens(segment_units(p, r.library)) for p in ctx])[:PROMPT_P]] * 4,
                           dtype=torch.long, device=r.dev)
    out = r.adapter.generate(prompts, LEN_P - PROMPT_P, 1.0, r.grammar, "def", rng=r.trng)
    n_valid = 0
    for toks in out[:, PROMPT_P:].tolist():
        body = toks[:toks.index(EDEF)] if EDEF in toks else None
        pat = parse_def(body, r.library) if body is not None else None
        if pat is not None:
            n_valid += 1
            rew, dj, sup, ldef, fp = candidate_reward(pat, sc, r.library, set())
            assert -1.0 <= rew <= 5.0
    assert n_valid == 4, "grammar-constrained DEFs must always parse"
    gen_mask = torch.zeros_like(out, dtype=torch.bool)
    gen_mask[:, PROMPT_P:] = out[:, PROMPT_P:] != PAD
    r.adapter.train()
    lp = r.adapter.seq_logprob(out, gen_mask, r.grammar, "def", PROMPT_P)
    assert lp.shape == (4,) and torch.isfinite(lp).all() and lp.requires_grad and (lp <= 0).all()
    for fh in r.logs.values():
        fh.close()
