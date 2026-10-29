"""Tests of config: every preset is consistent. Run from experiments/symbolics:
    ../../venv312/Scripts/python.exe -m pytest arc_agent_v2/tests/test_config.py -q
"""
import pytest

from arc_agent_v2.config.config import PRESETS, BrainConfig, Region


def test_presets_are_consistent():
    for name, cfg in PRESETS.items():
        order = cfg.validate()
        assert len(order) == len(cfg.regions), name
        seen = set(cfg.sensors)
        for r in order:                                 # each region comes after its feedforward sources
            assert set(r.feedforward) <= seen
            seen.add(r.name)
    assert PRESETS["arithmetic"].sdr.d == 2048


def test_validate_rejects_bad_configs():
    with pytest.raises(ValueError):
        BrainConfig(sensors=("s",), regions=(Region("A", 1, "learned", ("B",)), Region("B", 1, "learned", ("A",))
                                             )).validate()
    with pytest.raises(ValueError):
        BrainConfig(sensors=("s",), regions=(Region("A", 1, "learned", ("nowhere",)),)).validate()
    with pytest.raises(ValueError):
        BrainConfig(regions=(Region("A", 1, "magic"),)).validate()
