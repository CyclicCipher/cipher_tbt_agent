"""
Run the math classroom with tracing enabled.

Captures the activation flow for each test question and saves to
trace.json for the HTML visualiser.

Usage:
    python experiments/symbolic_ai_v2/ctkg/logic/trace_runner.py
"""
import sys
import os
import random

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.symbolic_ai_v2.environments.math_classroom import MathClassroomEnv
from experiments.symbolic_ai_v2.ctkg.logic.graph import (
    KnowledgeGraph, COOCCURRENCE, TRANSITION, ACTIVATION_THRESHOLD,
)
from experiments.symbolic_ai_v2.ctkg.logic.loop import AgenticLoop
from experiments.symbolic_ai_v2.ctkg.logic.trace import Tracer
import json


def run_traced(max_cycles=5, counting_warmup=2, seed=43):
    env = MathClassroomEnv(
        problem_type="succession", mode="A", max_cycles=max_cycles,
        counting_warmup=counting_warmup, answer_range=(0, 9),
        train_per_cycle=3, test_per_cycle=3, seed=seed,
    )
    kg = KnowledgeGraph()
    loop = AgenticLoop(kg)
    loop.CONSOLIDATION_INTERVAL = 0
    loop.set_preferred("FEEDBACK_correct", +1.0)
    loop.set_preferred("FEEDBACK_wrong", -1.0)
    rng = random.Random(seed)

    tracer = Tracer()
    tracer.enabled = True
    records = []

    step = 0
    while not env.done and step < 2000:
        obs = env.observe()
        tokens = [t[0] for t in obs]
        etypes = [t[1] for t in obs]

        # Is this a test question where the agent needs to pick a digit?
        is_test_digit = (
            env._phase == "test"
            and env._waiting_for_answer
            and len(env._answer_buffer) == 0
        )

        if is_test_digit:
            correct = env._correct_answer
            tracer.start_record(step=step, correct_answer=correct)

            # --- Phase 1: capture activations BEFORE observe ---
            before_acts = {
                nid: n.activation for nid, n in kg._nodes.items()
            }

        # Run observe
        loop.observe(tokens, etypes)

        if is_test_digit:
            # --- Phase 1 cont: capture observation tokens ---
            current_nids = [kg.get_or_create(t) for t in tokens]
            tracer.capture_observation(kg, current_nids)

            # --- Phase 2: capture co-occurrence spread result ---
            after_acts = {
                nid: n.activation for nid, n in kg._nodes.items()
            }
            tracer.capture_cooccur(kg, before_acts, after_acts)

            # --- Phase 3: capture sigma state ---
            active_set = set(
                nid for nid, n in kg._nodes.items()
                if n.activation >= ACTIVATION_THRESHOLD
            )
            # Recompute Q·K for tracing (same as compute_sigma does internally)
            qk = {}
            active_acts = {
                nid: kg._nodes[nid].activation
                for nid in active_set if nid in kg._nodes
            }
            for tgt_nid in active_set:
                support = 0.0
                for edge in kg._incoming.get(tgt_nid, ()):
                    if edge.role != COOCCURRENCE:
                        continue
                    c_nid = edge.source
                    c_act = active_acts.get(c_nid, 0.0)
                    if c_act > 0 and edge._w > 0:
                        support += c_act * edge._w
                qk[tgt_nid] = support
            tracer.capture_sigma(kg, active_set, qk)

        # --- Act ---
        actions = env.available_actions()
        if not actions:
            break

        if is_test_digit:
            # --- Phase 4: capture forward/backward scores ---
            candidate_nids = [kg.get_or_create(a) for a in actions]
            ctx = dict(loop._last_actual) if loop._last_actual else {}
            candidate_set = set(candidate_nids)
            for nid in candidate_set:
                ctx.pop(nid, None)

            # Forward spread (copy of select_action logic)
            fwd = {}
            for src_nid, src_act in ctx.items():
                if src_act <= 0:
                    continue
                for edge in kg._outgoing.get(src_nid, ()):
                    ew = edge.effective_weight
                    if ew <= 0:
                        continue
                    tgt = edge.target
                    if tgt in candidate_set:
                        fwd[tgt] = fwd.get(tgt, 0.0) + ew * src_act
                    elif tgt not in ctx:
                        for edge2 in kg._outgoing.get(tgt, ()):
                            ew2 = edge2.effective_weight
                            if ew2 <= 0:
                                continue
                            if edge2.target in candidate_set:
                                fwd[edge2.target] = (
                                    fwd.get(edge2.target, 0.0) + ew2 * ew * src_act
                                )

            # Backward spread
            bwd = {}
            preferred = kg.preferred_nodes()
            for pref_nid, pref_val in preferred.items():
                if pref_val <= 0:
                    continue
                for edge in kg._incoming.get(pref_nid, ()):
                    ew = edge.effective_weight
                    if ew <= 0:
                        continue
                    src = edge.source
                    if src in candidate_set:
                        bwd[src] = bwd.get(src, 0.0) + ew * pref_val
                    else:
                        for edge2 in kg._incoming.get(src, ()):
                            ew2 = edge2.effective_weight
                            if ew2 <= 0:
                                continue
                            if edge2.source in candidate_set:
                                bwd[edge2.source] = (
                                    bwd.get(edge2.source, 0.0) + ew2 * ew * pref_val
                                )

            # Normalise
            for signal in (fwd, bwd):
                if signal:
                    mx = max(signal.values())
                    if mx > 0:
                        for nid in signal:
                            signal[nid] /= mx

            tracer.capture_forward(kg, fwd, bwd, candidate_nids)

        chosen = loop.act(actions)
        if chosen is None:
            chosen = rng.choice(actions)

        if is_test_digit:
            tracer.capture_selection(chosen)

        loop.observe([chosen], [2])
        env.act(chosen)
        step += 1

    # Save trace
    out_path = os.path.join(os.path.dirname(__file__), "trace.json")
    tracer.save(out_path)
    print(f"Score: {env._total_correct}/{env._total_tested} = {env.score:.1%}")
    print(f"Traced {len(tracer.records)} test questions to {out_path}")
    return tracer


if __name__ == "__main__":
    run_traced()
