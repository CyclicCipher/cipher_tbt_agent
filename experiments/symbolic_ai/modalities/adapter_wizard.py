"""adapter_wizard.py -- Phase S.1b: Training, calibration, and verification tool.

Usage::

    python modalities/adapter_wizard.py --train [--output glyph_reader.pkl]
    python modalities/adapter_wizard.py --smoke
    python modalities/adapter_wizard.py --verify --window "TiTS"
    python modalities/adapter_wizard.py --info [--reader glyph_reader.pkl]
"""
from __future__ import annotations

import argparse
import sys
import os

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np

from modalities.glyph_reader import GlyphReader, _DEFAULT_CHARSET
from modalities.screen_reader import ScreenReader


def cmd_train(args: argparse.Namespace) -> None:
    """Train GlyphReader and save to disk."""
    print("=" * 60)
    print("  Adapter Wizard -- Training GlyphReader")
    print("=" * 60)
    print(f"  Characters:  {len(_DEFAULT_CHARSET)}")
    print(f"  Clusters:    {args.clusters}")
    print(f"  Output:      {args.output}")
    print()
    reader = GlyphReader(n_clusters=args.clusters, model_path=args.output)
    reader.train(verbose=True)
    reader.save(args.output)
    print()
    print(reader.summary())
    print()
    print(f"  Saved to: {args.output}")
    print("  PASS -- GlyphReader trained successfully.")


def cmd_info(args: argparse.Namespace) -> None:
    """Print a summary of a trained GlyphReader."""
    if not os.path.exists(args.reader):
        print(f"  {args.reader} not found -- run --train first")
        sys.exit(1)
    reader = GlyphReader.load(args.reader)
    print(reader.summary())
    print("\n  Cluster label map (top 20 by confidence):")
    sorted_clusters = sorted(reader._label_conf.items(), key=lambda x: -x[1])[:20]
    for cid, conf in sorted_clusters:
        ch = reader._labels.get(cid, "")
        print(f"    cluster {cid:4d}  -> {ch!r:10s}  conf={conf:.0%}")


def cmd_smoke_test(args: argparse.Namespace) -> None:
    """Train GlyphReader; read back test characters; report accuracy."""
    print("=" * 60)
    print("  Adapter Wizard -- GlyphReader Smoke Test (T0)")
    print("=" * 60)

    reader = GlyphReader(n_clusters=args.clusters, model_path=args.output)
    reader.train(verbose=True)

    test_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    correct = 0
    total   = 0
    errors  = []

    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore[import]
        font = ImageFont.load_default()
    except ImportError:
        print("  PIL not available -- cannot run smoke test")
        return

    for ch in test_chars:
        patch = reader._render_char(ch, font, augment=False)
        if patch is None:
            continue
        result = reader.read_patch(patch)
        total += 1
        if result.char == ch:
            correct += 1
        else:
            errors.append((ch, result.char, result.confidence))

    acc = correct / max(1, total)
    print(f"\n  Accuracy: {correct}/{total} = {acc:.0%}")
    if errors:
        print("  Errors (char -> read, conf): ", end="")
        print(", ".join(f"{c!r}->{r!r}({conf:.0%})" for c, r, conf in errors[:10]))
    passed = acc >= 0.70
    print(f"\n  {'PASS' if passed else 'FAIL'} -- smoke test complete.")
    if passed:
        reader.save(args.output)
        print(f"  Saved to: {args.output}")


def cmd_verify(args: argparse.Namespace) -> None:
    """Capture a frame from the target window and show what the agent reads."""
    print("=" * 60)
    print("  Adapter Wizard -- Verify Screen Reading")
    print("=" * 60)
    if not os.path.exists(args.reader):
        print(f"  {args.reader} not found -- run --train first")
        sys.exit(1)
    print(f"  Loading reader from {args.reader} ...")
    screen_reader = ScreenReader.load(args.reader)
    frame = _capture_window(args.window)
    if frame is None:
        print("  Could not capture window. Is the game running?")
        sys.exit(1)
    print(f"  Captured frame: {frame.shape[1]}x{frame.shape[0]} px")
    if args.calibrate:
        print("  Calibrating to game rendering ...")
        screen_reader.calibrate(frame, verbose=True)
    print("  Reading frame ...")
    reading = screen_reader.read(frame)
    print()
    print(f"  Detected {len(reading.raw_regions)} regions:")
    for r in reading.raw_regions:
        print(f"    [{r.region_type:12s}] ({r.x0},{r.y0})-({r.x1},{r.y1}) conf={r.confidence:.2f}")
    print()
    print(f"  Narrative ({len(reading.narrative)} chars):")
    for line in reading.narrative.split("\n")[:10]:
        print(f"    {line!r}")
    print()
    print(f"  Buttons ({len(reading.buttons)}):")
    for b in reading.buttons:
        print(f"    {b!r}")
    print()
    print(f"  Stats text: {reading.stats_text!r}")


def _capture_window(window_title: str):
    """Capture the window with the given title using mss or PIL."""
    try:
        import mss  # type: ignore[import]
        try:
            import ctypes
            hwnd = ctypes.windll.user32.FindWindowW(None, window_title)  # type: ignore[attr-defined]
            if hwnd:
                rect = ctypes.wintypes.RECT()  # type: ignore[attr-defined]
                ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect))  # type: ignore[attr-defined]
                mon = {"left": rect.left, "top": rect.top,
                       "width": rect.right - rect.left,
                       "height": rect.bottom - rect.top}
                with mss.mss() as sct:
                    raw = sct.grab(mon)
                    arr = np.frombuffer(raw.raw, dtype=np.uint8)
                    arr = arr.reshape((raw.height, raw.width, 4))
                    return arr[:, :, :3]
        except Exception:
            pass
        with mss.mss() as sct:
            raw = sct.grab(sct.monitors[1])
            arr = np.frombuffer(raw.raw, dtype=np.uint8)
            arr = arr.reshape((raw.height, raw.width, 4))
            return arr[:, :, :3]
    except ImportError:
        pass
    try:
        from PIL import ImageGrab  # type: ignore[import]
        img = ImageGrab.grab()
        return np.array(img)[:, :, :3]
    except Exception:
        pass
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adapter Wizard -- train, calibrate, and verify the ScreenReader"
    )
    parser.add_argument("--train",    action="store_true", help="Train GlyphReader")
    parser.add_argument("--verify",   action="store_true", help="Verify reading on live frame")
    parser.add_argument("--info",     action="store_true", help="Print GlyphReader summary")
    parser.add_argument("--smoke",    action="store_true", help="Run T0 smoke test")
    parser.add_argument("--window",   default="TiTS",      help="Window title to capture")
    parser.add_argument("--reader",   default="glyph_reader.pkl", help="Reader model path")
    parser.add_argument("--output",   default="glyph_reader.pkl", help="Output path for --train")
    parser.add_argument("--clusters", type=int, default=128, help="Number of glyph clusters")
    parser.add_argument("--calibrate", action="store_true",
                        help="Calibrate reader to game rendering during --verify")
    args = parser.parse_args()

    if args.train:
        cmd_train(args)
    elif args.verify:
        cmd_verify(args)
    elif args.info:
        cmd_info(args)
    elif args.smoke:
        cmd_smoke_test(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
