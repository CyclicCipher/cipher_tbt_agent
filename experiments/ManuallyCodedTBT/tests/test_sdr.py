"""
test_sdr.py — Tests for the SDR library.

Run from project root:
    python experiments/ManuallyCodedTBT/tests/test_sdr.py

Tests cover:
    1. Creation and basic properties
    2. Bitwise operations
    3. Overlap and matching
    4. Union operations and saturation
    5. Subsampling
    6. Concatenation and splitting
    7. Noise resistance
    8. Scalar encoding (semantic overlap)
    9. Periodic encoding (wrap-around)
    10. RDSE (random distributed scalar encoding)
    11. Multi-encoding
    12. False positive probability
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
from sdr import (
    sdr_random, sdr_from_indices, sdr_empty, active_indices, population,
    sdr_and, sdr_or, sdr_xor, sdr_not,
    overlap, match, overlap_score_normalized,
    union, union_saturation, match_union,
    subsample, concatenate, split,
    add_noise, encode_scalar, decode_scalar,
    encode_periodic, RDSEncoder, encode_multi,
    capacity, false_positive_probability,
    sdr_to_string, sdr_density_bar,
)


passed = 0
failed = 0


def test(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✓ {name}")
    else:
        failed += 1
        msg = f"  ✗ {name}"
        if detail:
            msg += f"  — {detail}"
        print(msg)


def section(name: str):
    print(f"\n{'─' * 60}")
    print(f"  {name}")
    print(f"{'─' * 60}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. Creation
# ══════════════════════════════════════════════════════════════════════════════
section("1. Creation and basic properties")

rng = np.random.default_rng(42)
a = sdr_random(2048, 40, rng)
test("sdr_random length", len(a) == 2048)
test("sdr_random population", population(a) == 40)
test("sdr_random dtype", a.dtype == bool)

b = sdr_from_indices(100, np.array([0, 10, 50, 99]))
test("sdr_from_indices population", population(b) == 4)
test("sdr_from_indices correct bits", b[0] and b[10] and b[50] and b[99])
test("sdr_from_indices others off", not b[1] and not b[49])

e = sdr_empty(256)
test("sdr_empty population", population(e) == 0)

indices = active_indices(b)
test("active_indices correct", np.array_equal(indices, [0, 10, 50, 99]))

# ══════════════════════════════════════════════════════════════════════════════
# 2. Bitwise operations
# ══════════════════════════════════════════════════════════════════════════════
section("2. Bitwise operations")

x = sdr_from_indices(10, np.array([0, 1, 2, 3]))
y = sdr_from_indices(10, np.array([2, 3, 4, 5]))

and_xy = sdr_and(x, y)
test("AND correct", np.array_equal(active_indices(and_xy), [2, 3]))

or_xy = sdr_or(x, y)
test("OR correct", np.array_equal(active_indices(or_xy), [0, 1, 2, 3, 4, 5]))

xor_xy = sdr_xor(x, y)
test("XOR correct", np.array_equal(active_indices(xor_xy), [0, 1, 4, 5]))

not_x = sdr_not(x)
test("NOT population", population(not_x) == 6)
test("NOT flips correctly", not not_x[0] and not_x[4])

# ══════════════════════════════════════════════════════════════════════════════
# 3. Overlap and matching
# ══════════════════════════════════════════════════════════════════════════════
section("3. Overlap and matching")

test("overlap correct", overlap(x, y) == 2)
test("overlap with self", overlap(x, x) == population(x))
test("overlap with empty", overlap(x, sdr_empty(10)) == 0)

test("match theta=2", match(x, y, theta=2))
test("match theta=3 fails", not match(x, y, theta=3))

norm = overlap_score_normalized(x, y)
test("normalized overlap", abs(norm - 0.5) < 0.01, f"got {norm}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. Union operations
# ══════════════════════════════════════════════════════════════════════════════
section("4. Union operations and saturation")

rng2 = np.random.default_rng(100)
sdrs = [sdr_random(256, 10, rng2) for _ in range(5)]
u = union(sdrs)
test("union length preserved", len(u) == 256)
test("union population >= max component",
     population(u) >= max(population(s) for s in sdrs))

# Every component should match the union
for i, s in enumerate(sdrs):
    test(f"component {i} matches union", match_union(s, u, theta=8))

# Saturation check
sat = union_saturation(u)
test("saturation reasonable", 0.0 < sat < 1.0, f"got {sat:.3f}")

# Saturation increases with more SDRs
big_union = union([sdr_random(256, 10, rng2) for _ in range(50)])
big_sat = union_saturation(big_union)
test("more SDRs = higher saturation", big_sat > sat,
     f"50 SDRs: {big_sat:.3f} vs 5 SDRs: {sat:.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. Subsampling
# ══════════════════════════════════════════════════════════════════════════════
section("5. Subsampling")

rng3 = np.random.default_rng(200)
full = sdr_random(2048, 40, rng3)
sub = subsample(full, 10, rng3)
test("subsampled population", population(sub) == 10)
test("subsampled bits are subset", overlap(sub, full) == 10)

# Subsampled should still match the original with lowered theta
test("subsampled matches original", match(sub, full, theta=8))

# ══════════════════════════════════════════════════════════════════════════════
# 6. Concatenation and splitting
# ══════════════════════════════════════════════════════════════════════════════
section("6. Concatenation and splitting")

p1 = sdr_random(100, 5, np.random.default_rng(1))
p2 = sdr_random(200, 10, np.random.default_rng(2))
p3 = sdr_random(50, 3, np.random.default_rng(3))

cat = concatenate([p1, p2, p3])
test("concatenated length", len(cat) == 350)
test("concatenated population", population(cat) == 18)

parts = split(cat, [100, 200, 50])
test("split count", len(parts) == 3)
test("split part 1 matches", np.array_equal(parts[0], p1))
test("split part 2 matches", np.array_equal(parts[1], p2))
test("split part 3 matches", np.array_equal(parts[2], p3))

# ══════════════════════════════════════════════════════════════════════════════
# 7. Noise resistance
# ══════════════════════════════════════════════════════════════════════════════
section("7. Noise resistance")

rng4 = np.random.default_rng(42)
clean = sdr_random(2048, 40, rng4)

noisy = add_noise(clean, num_flips=5, rng=np.random.default_rng(99))
test("noisy same population", population(noisy) == population(clean))
test("noisy differs from clean", not np.array_equal(noisy, clean))

ov = overlap(clean, noisy)
test("noisy overlap high", ov == 35, f"expected 35, got {ov}")

# Even with 10 bits flipped, still high overlap
very_noisy = add_noise(clean, num_flips=10, rng=np.random.default_rng(99))
ov2 = overlap(clean, very_noisy)
test("very noisy overlap >= 30", ov2 >= 30, f"got {ov2}")
test("very noisy still matches theta=20", match(clean, very_noisy, theta=20))

# ══════════════════════════════════════════════════════════════════════════════
# 8. Scalar encoding
# ══════════════════════════════════════════════════════════════════════════════
section("8. Scalar encoding — semantic overlap")

# Use n=256, w=21, range 0-20 so each unit moves the bucket ~12 positions
# and w=21 means adjacent values have ~9 bits of overlap
n, w = 256, 21

enc_5 = encode_scalar(5, n, w, min_val=0, max_val=20)
enc_6 = encode_scalar(6, n, w, min_val=0, max_val=20)
enc_10 = encode_scalar(10, n, w, min_val=0, max_val=20)
enc_18 = encode_scalar(18, n, w, min_val=0, max_val=20)

test("scalar encoding population", population(enc_5) == w)

# Adjacent values should have high overlap
ov_adjacent = overlap(enc_5, enc_6)
test("adjacent values high overlap", ov_adjacent >= w // 2,
     f"5↔6 overlap = {ov_adjacent}")

# Distant values should have low/no overlap
ov_distant = overlap(enc_5, enc_18)
test("distant values low overlap", ov_distant <= 3,
     f"5↔18 overlap = {ov_distant}")

# Decode should recover approximately
decoded = decode_scalar(enc_5, w, min_val=0, max_val=20)
test("decode scalar ~5", abs(decoded - 5) < 1.5, f"decoded {decoded:.1f}")

decoded_10 = decode_scalar(enc_10, w, min_val=0, max_val=20)
test("decode scalar ~10", abs(decoded_10 - 10) < 1.5, f"decoded {decoded_10:.1f}")

# Monotonic overlap: closer values = more overlap
# With n=256, w=21, range 0-10: step=23.5px, bucket=21px → no overlap.
# Use w=51 so the bucket is wider than the step.
n2, w2 = 256, 51
enc_a = encode_scalar(3, n2, w2, 0, 10)
enc_b = encode_scalar(4, n2, w2, 0, 10)
enc_c = encode_scalar(6, n2, w2, 0, 10)
ov_close = overlap(enc_a, enc_b)
ov_far = overlap(enc_a, enc_c)
test("closer = more overlap", ov_close > ov_far,
     f"3↔4={ov_close}, 3↔6={ov_far}")

# ══════════════════════════════════════════════════════════════════════════════
# 9. Periodic encoding
# ══════════════════════════════════════════════════════════════════════════════
section("9. Periodic encoding — wrap-around")

# n=256, w=21, range 0-24: each hour = ~10.67 positions, w=21 covers ~2 hours
n_p, w_p = 256, 21

enc_0h = encode_periodic(0, n_p, w_p, min_val=0, max_val=24)
enc_23h = encode_periodic(23, n_p, w_p, min_val=0, max_val=24)
enc_1h = encode_periodic(1, n_p, w_p, min_val=0, max_val=24)
enc_12h = encode_periodic(12, n_p, w_p, min_val=0, max_val=24)

test("periodic population", population(enc_0h) == w_p)

# 0h and 23h should be close (1 hour apart, wrap-around)
ov_wrap = overlap(enc_0h, enc_23h)
test("wrap-around: 0h↔23h close", ov_wrap >= w_p // 2,
     f"overlap = {ov_wrap}")

# 0h and 1h should also be close (sanity)
ov_adj = overlap(enc_0h, enc_1h)
test("adjacent: 0h↔1h close", ov_adj >= w_p // 2,
     f"overlap = {ov_adj}")

# 0h and 12h should be far (opposite side of the cycle)
ov_far = overlap(enc_0h, enc_12h)
test("opposite: 0h↔12h far", ov_far <= 3,
     f"overlap = {ov_far}")

# ══════════════════════════════════════════════════════════════════════════════
# 10. RDSE (Random Distributed Scalar Encoding)
# ══════════════════════════════════════════════════════════════════════════════
section("10. RDSE — Random Distributed Scalar Encoding")

rdse = RDSEncoder(n=256, w=21, resolution=1.0, seed=42)

r_10 = rdse.encode(10)
r_11 = rdse.encode(11)
r_15 = rdse.encode(15)
r_50 = rdse.encode(50)

test("RDSE population", population(r_10) == 21)
test("RDSE deterministic", np.array_equal(r_10, rdse.encode(10)))

# Adjacent values differ by exactly 1 bit
ov_rdse_adj = overlap(r_10, r_11)
test("RDSE adjacent overlap = w-1", ov_rdse_adj == 20,
     f"got {ov_rdse_adj}")

# Further values have less overlap
ov_rdse_5 = overlap(r_10, r_15)
test("RDSE 5-apart overlap = w-5", ov_rdse_5 == 16,
     f"got {ov_rdse_5}")

# Distant values have very low overlap
ov_rdse_far = overlap(r_10, r_50)
test("RDSE distant low overlap", ov_rdse_far <= 5,
     f"got {ov_rdse_far}")

# Decode
decoded_rdse = rdse.decode(r_10)
test("RDSE decode ~10", decoded_rdse is not None and abs(decoded_rdse - 10) < 1.5,
     f"decoded {decoded_rdse}")

# ══════════════════════════════════════════════════════════════════════════════
# 11. Multi-encoding
# ══════════════════════════════════════════════════════════════════════════════
section("11. Multi-encoding")

configs = [
    {'n': 128, 'w': 11, 'min_val': 0, 'max_val': 6, 'periodic': True},  # day of week
    {'n': 128, 'w': 11, 'min_val': 0, 'max_val': 24, 'periodic': True},  # hour
    {'n': 64, 'w': 7, 'min_val': 0, 'max_val': 3, 'periodic': True},    # season
]

monday_9am_summer = encode_multi([0, 9, 1], configs)
monday_10am_summer = encode_multi([0, 10, 1], configs)
friday_9am_winter = encode_multi([4, 9, 3], configs)

test("multi length", len(monday_9am_summer) == 320)
test("multi population", population(monday_9am_summer) == 29)

# Same day + close hour + same season should have high overlap
ov_close = overlap(monday_9am_summer, monday_10am_summer)
# Different day + same hour + different season should have lower overlap
ov_diff = overlap(monday_9am_summer, friday_9am_winter)
test("multi: similar > different", ov_close > ov_diff,
     f"close={ov_close}, diff={ov_diff}")

# ══════════════════════════════════════════════════════════════════════════════
# 12. False positive probability
# ══════════════════════════════════════════════════════════════════════════════
section("12. False positive probability")

p_single = false_positive_probability(n=2048, w=40, theta=10)
test("FP probability very low", p_single < 1e-6,
     f"got {p_single:.2e}")

p_large_set = false_positive_probability(n=2048, w=40, theta=10,
                                          num_sdrs_in_set=10000)
test("FP with 10k SDRs still low", p_large_set < 0.01,
     f"got {p_large_set:.6f}")

# Higher theta = lower FP
p_strict = false_positive_probability(n=2048, w=40, theta=15)
test("stricter theta = lower FP", p_strict < p_single,
     f"theta=15: {p_strict:.2e} vs theta=10: {p_single:.2e}")

# Larger SDRs = lower FP
p_big = false_positive_probability(n=4096, w=40, theta=10)
test("larger SDR = lower FP", p_big < p_single,
     f"n=4096: {p_big:.2e} vs n=2048: {p_single:.2e}")

# ══════════════════════════════════════════════════════════════════════════════
# Display utilities demo
# ══════════════════════════════════════════════════════════════════════════════
section("Display utilities demo")

demo_sdr = encode_scalar(25, n=256, w=21, min_val=0, max_val=100)
print(f"  {sdr_to_string(demo_sdr)}")
print(f"  density: |{sdr_density_bar(demo_sdr, 64)}|")

demo_sdr2 = encode_scalar(75, n=256, w=21, min_val=0, max_val=100)
print(f"  {sdr_to_string(demo_sdr2)}")
print(f"  density: |{sdr_density_bar(demo_sdr2, 64)}|")


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 60}")
print(f"  RESULTS: {passed} passed, {failed} failed, {passed + failed} total")
print(f"{'═' * 60}")

if failed > 0:
    sys.exit(1)