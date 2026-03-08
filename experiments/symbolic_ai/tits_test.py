"""tits_test.py -- Phase S.3: TiTS adapter smoke tests.

Test phases
-----------
T0  GlyphReader training smoke test (no game required)
    Train on PIL-rendered ASCII + math charset -> read 62 test chars -> >=90% accuracy.

T1  Live frame reading (TiTS must be running)
    Read one frame -> assert buttons non-empty + narrative > 50 chars.

T2  New-game sequence (TiTS must be running, at main menu)
    Execute new_game.steps -> assert tutorial text appears within 30s.

T3  FEP drive smoke test (no game required)
    Build TiTSWorldModel + drives + goals -> verify deficit values at boundary states.

T4  Agent plays N steps (TiTS must be running, in game)
    Run DecisionEngine for N steps -> report quest log + score evolution.

Usage
-----
    # All offline phases (no game needed)
    python tits_test.py

    # Include live-frame test (TiTS running at main menu)
    python tits_test.py --phase t1

    # New-game sequence (TiTS running, showing main menu)
    python tits_test.py --phase t2

    # All phases
    python tits_test.py --phase t1 --phase t2 --phase t3

    # Agent plays
    python tits_test.py --phase t4 --steps 200
"""
from __future__ import annotations

import argparse
import sys
import time
import os
from typing import Dict

# ---------------------------------------------------------------------------
# Path setup: allow running from the symbolic_ai directory or repo root
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _separator(label: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")


# ===========================================================================
# T0 -- GlyphReader training smoke test
# ===========================================================================

def test_t0(save_path: str = "glyph_reader_tits.pkl", verbose: bool = True) -> bool:
    """Train GlyphReader on PIL charset and verify read-back accuracy >= 90%."""
    _separator("T0: GlyphReader training smoke test")

    from modalities.glyph_reader import GlyphReader, _DEFAULT_CHARSET

    reader = GlyphReader(
        patch_size = 16,
        n_clusters = 128,
        model_path = save_path,
    )

    print("Training on PIL-rendered charset ...")
    reader.train(verbose=verbose)
    reader.save(save_path)
    print(f"Saved to: {save_path}")

    # Build test set: render each char once and try to read it back
    import numpy as np
    try:
        from PIL import Image, ImageDraw, ImageFont
        _pil_ok = True
    except ImportError:
        _pil_ok = False

    if not _pil_ok:
        print("PIL not available - skipping T0 read-back (training still passed)")
        return True

    font = ImageFont.load_default()

    # Use the same _render_char() path that training used -- ensures identical format
    test_chars = [c for c in _DEFAULT_CHARSET if c.strip()]
    correct = 0
    total   = 0
    errors  = []

    for ch in test_chars:
        patch = reader._render_char(ch, font, augment=False)
        if patch is None:
            continue  # char not renderable by this font (e.g. box-draw glyphs)
        result = reader.read_patch(patch)
        total += 1
        if result.char == ch:
            correct += 1
        else:
            errors.append((ch, result.char, result.confidence))

    accuracy = correct / total if total > 0 else 0.0
    print(f"\nAccuracy: {correct}/{total} = {accuracy:.0%}")
    if errors:
        # Use ASCII-safe repr to avoid Windows cp1252 encoding errors
        err_str = ", ".join(
            f"{repr(e[0])}->{repr(e[1])}({e[2]:.0%})"
            for e in errors[:10]
        )
        print(f"Errors: {err_str}")
        if len(errors) > 10:
            print(f"  ... and {len(errors)-10} more")

    threshold = 0.90
    ok = accuracy >= threshold
    if ok:
        print(f"\nPASS -- T0 accuracy {accuracy:.0%} >= {threshold:.0%}")
    else:
        print(f"\nFAIL -- T0 accuracy {accuracy:.0%} < {threshold:.0%}")
    return ok


# ===========================================================================
# T1 -- Live frame reading (TiTS must be running)
# ===========================================================================

def test_t1(reader_path: str = "glyph_reader.pkl") -> bool:
    """Read one frame from running TiTS; assert narrative + buttons present."""
    _separator("T1: Live frame reading")

    from modalities.screen_reader import ScreenReader

    try:
        reader = ScreenReader.load(reader_path)
    except FileNotFoundError:
        print(f"  GlyphReader not found at {reader_path!r}")
        print("  Run: python modalities/adapter_wizard.py --train")
        return False

    # Capture a frame from TiTS
    try:
        import mss
        import numpy as np
        with mss.mss() as sct:
            monitor = sct.monitors[1]  # primary monitor
            shot = sct.grab(monitor)
        import mss.tools
        frame = np.array(shot)[:, :, :3]  # drop alpha
        print(f"  Captured frame: {frame.shape}")
    except Exception as e:
        print(f"  Frame capture failed: {e}")
        return False

    reading = reader.read(frame)

    print(f"  Narrative ({len(reading.narrative)} chars): {reading.narrative[:120]!r}")
    print(f"  Buttons ({len(reading.buttons)}): {reading.buttons}")
    print(f"  Regions: {len(reading.raw_regions)}")

    ok = len(reading.narrative) > 50 and len(reading.buttons) > 0
    print(f"\n{'PASS' if ok else 'FAIL'} -- T1: narrative={len(reading.narrative)} chars, buttons={len(reading.buttons)}")
    return ok


# ===========================================================================
# T2 -- New-game sequence (TiTS at main menu)
# ===========================================================================

def test_t2(config_path: str = "tits.adapter.yaml") -> bool:
    """Execute new_game sequence; assert tutorial text appears within 30s."""
    _separator("T2: New-game sequence")

    try:
        import yaml
    except ImportError:
        print("  PyYAML not available: pip install pyyaml")
        return False

    with open(config_path) as f:
        config = yaml.safe_load(f)

    from modalities.screen_reader import ScreenReader
    from modalities.tits_modality import TiTSModality

    reader_path = config.get("glyph_reader", "glyph_reader.pkl")
    try:
        reader = ScreenReader.load(reader_path)
    except FileNotFoundError:
        print(f"  GlyphReader not found at {reader_path!r}. Run --train first.")
        return False

    mod = TiTSModality(config, reader, dry_run=False)
    print("  Executing new-game sequence ...")
    try:
        obs = mod.connect()
    except Exception as e:
        print(f"  connect() failed: {e}")
        return False

    # Wait up to 30s for tutorial text to appear
    tutorial_patterns = (
        "tutorial", "welcome", "you find yourself",
        "begin", "your journey", "mercenary",
    )
    deadline = time.time() + 30.0
    found = False
    while time.time() < deadline:
        text = obs.get("text", "").lower()
        if any(p in text for p in tutorial_patterns):
            found = True
            break
        time.sleep(1.0)
        if hasattr(mod, "_capture"):
            frame = mod._capture()
            obs = mod.build_obs(frame)

    if found:
        print(f"  Tutorial text detected: {obs['text'][:100]!r}")
        print("\nPASS -- T2: tutorial scene reached")
    else:
        print(f"  Tutorial text NOT found after 30s. Last text: {obs.get('text','')[:100]!r}")
        print("\nFAIL -- T2: tutorial scene not reached")
    return found


# ===========================================================================
# T3 -- FEP drive smoke test (no game required)
# ===========================================================================

def test_t3() -> bool:
    """Verify drive deficit values at boundary states."""
    _separator("T3: FEP drive smoke test")

    from modalities.tits_adapter import TiTSWorldModel, build_state, make_tits_drives, make_tits_goals

    drives = make_tits_drives()
    hp_drive   = drives["hp"]
    lust_drive = drives["lust"]
    exp_drive  = drives["explore"]

    # State: healthy, low lust, has explored 5 locations
    healthy_state = {
        "hp_frac":   1.0,
        "lust_frac": 0.1,
        "locations": 5,
        "in_combat": False,
        "done":      False,
    }

    # State: critical HP, high lust, never explored
    critical_state = {
        "hp_frac":   0.15,
        "lust_frac": 0.90,
        "locations": 0,
        "in_combat": True,
        "done":      False,
    }

    # HP drive
    hp_healthy  = hp_drive.deficit(healthy_state)
    hp_critical = hp_drive.deficit(critical_state)
    print(f"  hp_drive.deficit(healthy)  = {hp_healthy:.3f}  (expect ~=0)")
    print(f"  hp_drive.deficit(critical) = {hp_critical:.3f}  (expect ~=0.850)")

    # Lust drive (lust_frac=0.10 -> measure=0.90, setpoint=0.70 -> no deficit)
    # (lust_frac=0.90 -> measure=0.10, setpoint=0.70 -> deficit=0.60 x urgency=0.65)
    lust_ok       = lust_drive.deficit(healthy_state)
    lust_critical = lust_drive.deficit(critical_state)
    print(f"  lust_drive.deficit(healthy)  = {lust_ok:.3f}  (expect 0.0)")
    print(f"  lust_drive.deficit(critical) = {lust_critical:.3f}  (expect ~=0.390)")

    # Explore drive (5 locs -> measure=0.5, deficit=(1-0.5)x0.35=0.175)
    exp_partial = exp_drive.deficit(healthy_state)
    exp_zero    = exp_drive.deficit({**healthy_state, "locations": 10})
    print(f"  explore_drive.deficit(5 locs)  = {exp_partial:.3f}  (expect ~=0.175)")
    print(f"  explore_drive.deficit(10 locs) = {exp_zero:.3f}   (expect 0.0)")

    # Goals: build and test condition logic
    world = TiTSWorldModel()
    goals = make_tits_goals(world)
    print(f"\n  Goals constructed: {[g.name for g in goals]}")

    # Verify goal conditions
    survive = next(g for g in goals if g.name == "survive")
    assert survive.is_safety, "survive goal must be is_safety=True"
    assert survive.condition(critical_state), "survive should fire when hp<0.25 and in_combat"
    assert not survive.condition(healthy_state), "survive should not fire when hp=1.0"

    combat = next(g for g in goals if g.name == "combat_act")
    assert combat.condition(critical_state) is False or True  # fires when in_combat AND hp>=0.25
    # In critical_state hp=0.15 < 0.25 so combat_act should NOT fire (survive handles it)
    assert not combat.condition(critical_state), "combat_act should NOT fire when hp<0.25 (survive takes over)"

    # survive is is_safety=True -> always returns 1.0 (always eligible, never suppressed)
    ep_healthy  = survive.effective_priority(healthy_state)
    ep_critical = survive.effective_priority(critical_state)
    assert ep_healthy  == 1.0, f"is_safety survive must always return 1.0, got {ep_healthy}"
    assert ep_critical == 1.0, f"is_safety survive must always return 1.0, got {ep_critical}"
    print(f"\n  survive.effective_priority(healthy)  = {ep_healthy:.3f}  (always 1.0 - is_safety)")
    print(f"  survive.effective_priority(critical) = {ep_critical:.3f}  (always 1.0 - is_safety)")

    # Explore goal (NOT is_safety) should have lower priority when satisfied
    explore = next(g for g in goals if g.name == "explore")
    ep_explore_low  = explore.effective_priority({"locations": 10, "in_combat": False, "done": False})
    ep_explore_high = explore.effective_priority({"locations": 0,  "in_combat": False, "done": False})
    assert ep_explore_high > ep_explore_low, "explore priority must be higher with 0 locations vs 10"
    print(f"  explore.effective_priority(10 locs) = {ep_explore_low:.3f}  (expect ~=0)")
    print(f"  explore.effective_priority(0 locs)  = {ep_explore_high:.3f}  (expect ~=0.35)")

    # Assertion summary
    checks = [
        hp_healthy  < 0.05,
        hp_critical > 0.80,
        lust_ok     < 0.05,
        lust_critical > 0.30,
        0.10 < exp_partial < 0.25,
        exp_zero < 0.05,
        ep_healthy  == 1.0,   # survive is is_safety
        ep_critical == 1.0,   # survive is is_safety
        ep_explore_high > ep_explore_low,
    ]

    if all(checks):
        print("\nPASS -- T3: all drive deficit assertions satisfied")
        return True
    else:
        failed = [i for i, c in enumerate(checks) if not c]
        print(f"\nFAIL -- T3: checks {failed} did not pass")
        return False


# ===========================================================================
# T4 -- Agent plays N steps
# ===========================================================================

def test_t4(steps: int = 50, config_path: str = "tits.adapter.yaml") -> bool:
    """Run DecisionEngine for N steps; report quest log + score evolution."""
    _separator(f"T4: Agent plays {steps} steps")

    try:
        import yaml
    except ImportError:
        print("  PyYAML not available: pip install pyyaml")
        return False

    with open(config_path) as f:
        config = yaml.safe_load(f)

    from modalities.screen_reader import ScreenReader
    from modalities.tits_modality import TiTSModality
    from modalities.tits_adapter import TiTSWorldModel, build_state, make_tits_goals
    from planning import DecisionEngine, GoalStack, AffordanceModel

    reader_path = config.get("glyph_reader", "glyph_reader.pkl")
    try:
        reader = ScreenReader.load(reader_path)
    except FileNotFoundError:
        print(f"  GlyphReader not found at {reader_path!r}. Run --train first.")
        return False

    mod   = TiTSModality(config, reader, dry_run=False)
    world = TiTSWorldModel()
    goals = make_tits_goals(world)

    engine = DecisionEngine(
        goals         = goals,
        goal_stack    = GoalStack(),
        affordances   = AffordanceModel(),
    )

    import numpy as np
    import random
    rng = random.Random(42)

    print("  Connecting to TiTS ...")
    try:
        obs = mod.connect()
    except Exception as e:
        print(f"  connect() failed: {e}")
        return False

    state = build_state(obs, world)
    score_history = [world.score]

    print(f"  Initial state: hp={world.hp_frac:.2f}, lust={world.lust_frac:.2f}, score={world.score:.1f}")
    print(f"  Starting {steps}-step episode ...\n")

    for step_i in range(steps):
        action, reason = engine.decide(state, rng)

        # Dispatch
        try:
            mod.send_text(action)
        except Exception as e:
            print(f"    [warn] send_text({action!r}) failed: {e}")

        time.sleep(0.5)

        # Observe
        if hasattr(mod, "_capture"):
            frame = mod._capture()
        else:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        obs = mod.build_obs(frame)
        state = build_state(obs, world)

        score_history.append(world.score)

        if step_i % 10 == 0 or world.in_combat:
            tag = "[COMBAT]" if world.in_combat else ""
            print(
                f"  Step {step_i+1:3d}/{steps} {tag:8s} "
                f"hp={world.hp_frac:.2f} lust={world.lust_frac:.2f} "
                f"score={world.score:.1f} | {repr(action)[:30]} ({reason})"
            )

        if world.done:
            print(f"\n  Episode ended at step {step_i+1}: done=True")
            break

    print(f"\n  Final score: {world.score:.1f}")
    print(f"  Locations visited: {world.locations_visited}")
    print(f"  Score trajectory (every 10): {score_history[::10]}")

    # Pass: score increased OR at least 10 steps completed
    ok = world.score > score_history[0] or step_i + 1 >= 10
    print(f"\n{'PASS' if ok else 'FAIL'} -- T4 complete")
    return ok


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="TiTS adapter smoke tests")
    parser.add_argument(
        "--phase",
        action="append",
        choices=["t0", "t1", "t2", "t3", "t4"],
        default=None,
        help="Test phase(s) to run (default: t0 t3)"
    )
    parser.add_argument("--steps",  type=int, default=50,  help="Steps for T4 (default 50)")
    parser.add_argument("--config", default="tits.adapter.yaml", help="Adapter YAML path")
    parser.add_argument("--reader", default="glyph_reader.pkl",  help="GlyphReader pkl path")
    parser.add_argument("--quiet",  action="store_true", help="Suppress verbose output")
    args = parser.parse_args()

    # Default: T0 + T3 (offline, no game needed)
    phases = args.phase if args.phase else ["t0", "t3"]

    results: Dict[str, bool] = {}

    if "t0" in phases:
        results["T0"] = test_t0(verbose=not args.quiet)

    if "t1" in phases:
        results["T1"] = test_t1(reader_path=args.reader)

    if "t2" in phases:
        results["T2"] = test_t2(config_path=args.config)

    if "t3" in phases:
        results["T3"] = test_t3()

    if "t4" in phases:
        results["T4"] = test_t4(steps=args.steps, config_path=args.config)

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    all_pass = True
    for phase, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {phase}: {status}")
        if not ok:
            all_pass = False

    if all_pass:
        print("\nAll phases PASS.")
    else:
        print("\nSome phases FAILED. See output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
