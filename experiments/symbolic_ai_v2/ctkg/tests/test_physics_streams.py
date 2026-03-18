"""
Tests for ctkg/einstein/physics_streams.py.
Verifies that each stream produces physically reasonable observations.
These are NOT Einstein test tests — they verify stream correctness only.
"""
from __future__ import annotations

import math
import pytest

from experiments.symbolic_ai_v2.ctkg.einstein.physics_streams import (
    newtonian_mechanics_stream,
    em_wave_stream,
    michelson_morley_stream,
    mercury_precession_stream,
    lorentz_factor_stream,
    all_physics_streams,
    C_NATURAL,
)


class TestStreamBasic:

    def test_newtonian_has_3_sets(self):
        stream = newtonian_mechanics_stream()
        assert len(stream.observation_sets) == 3

    def test_newtonian_fma_correct(self):
        """F=ma set: output should equal force/mass for each observation."""
        stream = newtonian_mechanics_stream(seed=0)
        set1 = stream.observation_sets[0]
        for inp, out in set1:
            expected = inp["force"] / inp["mass"]
            assert abs(out - expected) < 1e-9

    def test_em_wave_speed_constant(self):
        """EM stream: all outputs should be ≈ c = 1.0."""
        stream = em_wave_stream(seed=0)
        for inp, out in stream.observation_sets[0]:
            assert abs(out - C_NATURAL) < 1e-3, f"wave speed {out} ≠ c={C_NATURAL}"

    def test_mm_null_result(self):
        """MM stream: all outputs should be ≈ 0."""
        stream = michelson_morley_stream(seed=0)
        for inp, out in stream.observation_sets[0]:
            assert abs(out) < 1e-4, f"fringe shift {out} not null"

    def test_mm_newtonian_prediction_nonzero(self):
        """MM stream newtonian_predictions should be > 0."""
        stream = michelson_morley_stream(seed=0)
        for inp, out in stream.newtonian_predictions[0]:
            assert out > 0.0, f"Newtonian fringe prediction {out} should be positive"

    def test_mercury_anomaly_near_43(self):
        """Mercury stream: outputs should be near 43 arcsec/century."""
        stream = mercury_precession_stream(seed=0)
        for inp, out in stream.observation_sets[0]:
            assert 40.0 < out < 46.0, f"Mercury anomaly {out} not near 43"

    def test_mercury_newtonian_zero(self):
        """Mercury Newtonian predictions should be 0."""
        stream = mercury_precession_stream(seed=0)
        for inp, out in stream.newtonian_predictions[0]:
            assert out == 0.0

    def test_lorentz_gamma_correct(self):
        """Lorentz stream: output should equal 1/sqrt(1-v^2/c^2)."""
        stream = lorentz_factor_stream(c=1.0, seed=0)
        for inp, out in stream.observation_sets[0]:
            v = inp["velocity"]
            expected = 1.0 / math.sqrt(1.0 - v * v)
            assert abs(out - expected) < 1e-9

    def test_all_streams_registry(self):
        streams = all_physics_streams(seed=0)
        assert len(streams) == 5
        names = {s.name for s in streams}
        assert "newtonian_mechanics" in names
        assert "em_wave" in names
        assert "michelson_morley" in names
        assert "mercury_precession" in names
        assert "lorentz_factor" in names
