"""Profile the Phase-1 BC pipeline — measure, don't assume.

Attributes wall-clock to phases (layout sampling, oracle rollout, materialize,
training, eval), then for the training loop breaks each step into true GPU compute
time (CUDA events) vs everything else (Python / kernel-launch / sync overhead), and
dumps a torch.profiler op table. Optionally A/Bs eager vs torch.compile to confirm
whether the loop is launch-overhead bound.

Run on the GPU:
    python -m agent.profile_bc --mechanic key_door --steps 200
CPU smoke:
    python -m agent.profile_bc --device cpu --steps 15 --no-compile --n-train 30 --n-test 20
"""

from __future__ import annotations

import argparse
import contextlib
import time
from typing import Callable, Dict, Optional

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

from agent.dataset import build_dataset, materialize
from agent.layouts import sample_layouts
from agent.trunk import build_model

torch.set_float32_matmul_precision("high")


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


@contextlib.contextmanager
def _timed(label: str, device: str, store: Dict[str, float]):
    _sync(device)
    t0 = time.perf_counter()
    yield
    _sync(device)
    store[label] = time.perf_counter() - t0


def _make_step(model, tensors, opt, scaler, device, bsz, use_amp) -> Callable[[], None]:
    frames, actions, targets = tensors
    n = targets.shape[0]

    def step() -> None:
        idx = torch.randint(0, n, (bsz,), device=device)
        fw, aw, tg = frames[idx].long(), actions[idx], targets[idx]
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits, _ = model(fw, aw)
            loss = F.cross_entropy(logits, tg)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    return step


def _throughput_ms(step: Callable[[], None], device: str, steps: int, warmup: int = 10) -> float:
    for _ in range(warmup):
        step()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(steps):
        step()
    _sync(device)
    return (time.perf_counter() - t0) / steps * 1000.0


def _gpu_ms_per_step(step: Callable[[], None], device: str, steps: int) -> Optional[float]:
    """Pure GPU compute time per step via CUDA events (cuda only)."""
    if device != "cuda":
        return None
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    total = 0.0
    for _ in range(steps):
        start.record()
        step()
        end.record()
        torch.cuda.synchronize()
        total += start.elapsed_time(end)
    return total / steps


def main() -> None:
    ap = argparse.ArgumentParser(description="Profile the Phase-1 BC pipeline")
    ap.add_argument("--mechanic", default="key_door")
    ap.add_argument("--binding", default="pope2d1")
    ap.add_argument("--n-train", type=int, default=250)
    ap.add_argument("--n-test", type=int, default=80)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--window", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-compile", action="store_true")
    args = ap.parse_args()
    device = args.device
    use_amp = device == "cuda"
    times: Dict[str, float] = {}

    print(f"device={device} binding={args.binding} batch={args.batch} steps={args.steps}\n")

    with _timed("1. sample_layouts (BFS-validated)", device, times):
        pool = sample_layouts(args.mechanic, args.n_train + args.n_test, seed=0)
    train_layouts, test_layouts = pool[: args.n_train], pool[args.n_train :]

    with _timed("2. build_dataset (oracle rollouts)", device, times):
        train_ds = build_dataset(train_layouts, window=args.window)
        test_ds = build_dataset(test_layouts, window=args.window)

    with _timed("3. materialize + to(device)", device, times):
        tensors = tuple(t.to(device) for t in materialize(train_ds))

    with _timed("4. build_model + warmup", device, times):
        model = build_model(args.binding, window=args.window).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        step = _make_step(model, tensors, opt, scaler, device,
                          min(args.batch, tensors[2].shape[0]), use_amp)
        step()

    print("=== phase timings (one-time setup) ===")
    for k, v in times.items():
        print(f"  {k:38s} {v*1000:8.1f} ms")
    print(f"  {'(N decisions)':38s} {tensors[2].shape[0]:8d}")

    ms_step = _throughput_ms(step, device, args.steps)
    gpu_ms = _gpu_ms_per_step(step, device, min(args.steps, 50))

    print("\n=== training loop ===")
    print(f"  wall time / step        {ms_step:8.2f} ms   (=> {ms_step*4000/1000:.0f}s for 4000 steps)")
    if gpu_ms is not None:
        overhead = ms_step - gpu_ms
        print(f"  pure GPU compute / step {gpu_ms:8.2f} ms")
        print(f"  non-GPU overhead / step {overhead:8.2f} ms   "
              f"({100*overhead/ms_step:.0f}% of wall time)")

    print("\n=== torch.profiler (30 steps) - top ops ===")
    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device == "cuda" else [])
    with profile(activities=activities) as prof:
        for _ in range(30):
            step()
        _sync(device)
    sort_key = "cuda_time_total" if device == "cuda" else "cpu_time_total"
    print(prof.key_averages().table(sort_by=sort_key, row_limit=15))

    print("\n=== config A/B (ms/step) ===")
    bsz = min(args.batch, tensors[2].shape[0])

    def time_cfg(mdl, amp_on: bool, warm: int) -> float:
        s = _make_step(mdl, tensors, opt, scaler, device, bsz, amp_on and device == "cuda")
        return _throughput_ms(s, device, args.steps, warmup=warm)

    rows = [("eager", "amp", time_cfg(model, True, 10)),
            ("eager", "fp32", time_cfg(model, False, 10))]
    if not args.no_compile and device == "cuda":
        try:
            cmodel = torch.compile(model, mode="reduce-overhead")
            rows.append(("compiled", "amp", time_cfg(cmodel, True, 25)))
            rows.append(("compiled", "fp32", time_cfg(cmodel, False, 25)))
        except Exception as e:  # noqa: BLE001
            print(f"  torch.compile unavailable: {e}")
    best = min(rows, key=lambda r: r[2])
    for mode, prec, ms in rows:
        tag = "  <- fastest" if (mode, prec) == best[:2] else ""
        print(f"  {mode:9s} {prec:4s} {ms:7.2f} ms/step{tag}")
    print(f"  best {best[0]}/{best[1]}: {best[2] * 4000 / 1000:.0f}s per arm, "
          f"{4 * best[2] * 4000 / 1000:.0f}s for the 4-arm sweep "
          f"(eager/amp was {ms_step * 4 * 4000 / 1000:.0f}s)")

    print("\n=== verdict ===")
    if gpu_ms is not None:
        if ms_step > 2 * gpu_ms:
            print(f"  LAUNCH/CPU-BOUND: GPU does {gpu_ms:.2f} ms of work but each step takes "
                  f"{ms_step:.2f} ms.\n  The model is tiny; per-step Python + kernel-launch "
                  f"overhead dominates and the GPU idles between launches.")
            if ms_compiled and ms_compiled < ms_step:
                print(f"  torch.compile gives {ms_step/ms_compiled:.1f}x — confirms launch-bound. "
                      f"Fix: compile the model (CUDA graphs) and/or raise --batch.")
        else:
            print(f"  GPU-BOUND: step {ms_step:.2f} ms ~ GPU {gpu_ms:.2f} ms. "
                  f"Speed comes from a bigger/faster model path, not launch overhead.")
    else:
        print("  (CPU run - re-run with --device cuda for the GPU-vs-overhead split.)")


if __name__ == "__main__":
    main()
