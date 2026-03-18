"""
Physics Observation Streams — Phase 9 prerequisites.

THIS MODULE IS NOT THE EINSTEIN TEST.

Implements four physically-grounded observation stream generators for the
real Einstein test (Phase 9). All streams use natural units (c = 1.0 unless
specified), anonymous operator symbols, and are seeded for reproducibility.

Unlike ctkg/einstein/streams.py (which generates synthetic linear analogs for
abduction routing tests), this module generates streams encoding actual physics:
  - Stream 1: Newtonian kinematics/dynamics at v ≪ c
  - Stream 2: Electromagnetic wave propagation (c as a universal constant)
  - Stream 3: Michelson-Morley null result (all fringe shifts = 0)
  - Stream 4: Mercury perihelion precession residual (GR anomaly)

All observation sets are lists of (input_bindings: dict[str, float], output: float).
Variable names in input_bindings are role labels, not physical quantity names —
they are just string keys in the dict and are NOT subject to the Iron Law.
The Iron Law applies only to operator NodeIds in SchematicLaw / Expr heads.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Constants (natural units: c = 1.0)
# ---------------------------------------------------------------------------

C_NATURAL = 1.0          # speed of light in natural units
G_NATURAL = 6.674e-11    # gravitational constant (SI, used for Mercury stream)
_MERCURY_GR_ARCSEC_PER_CENTURY = 43.0  # observed GR residual in arcsec/century


# ---------------------------------------------------------------------------
# Stream descriptor
# ---------------------------------------------------------------------------

@dataclass
class PhysicsStream:
    """A physically-grounded observation stream for the Einstein test.

    Attributes
    ----------
    name              : short identifier.
    description       : one-line description.
    observation_sets  : K lists of (input_bindings, observed_output) pairs.
    newtonian_predictions : corresponding Newtonian predictions for comparison.
                          None if Newtonian theory does not apply.
    c_value           : speed of light value used in this stream.
    expected_law      : human-readable string of the law to discover (metadata).
    """
    name:                    str
    description:             str
    observation_sets:        list[list[tuple[dict, float]]]
    newtonian_predictions:   list[list[tuple[dict, float]]] = field(default_factory=list)
    c_value:                 float = C_NATURAL
    expected_law:            str = ""


# ---------------------------------------------------------------------------
# Stream 1: Newtonian mechanics
# ---------------------------------------------------------------------------

def newtonian_mechanics_stream(n_per_law: int = 12, seed: int = 0) -> PhysicsStream:
    """Stream 1: Newtonian kinematics and dynamics at v ≪ c.

    Three observation sets:
      Set 1 — F = m*a (acceleration law):
        input:  {"force": F, "mass": m}
        output: F / m  (acceleration)

      Set 2 — Galilean transform (x' = x - v*t):
        input:  {"x": x, "v_frame": v, "t": t}
        output: x - v * t

      Set 3 — Momentum (p = m*v):
        input:  {"mass": m, "velocity": v}
        output: m * v

    All velocities v_frame ≪ c = 1.0.
    """
    rng = random.Random(seed)

    # Set 1: F=ma → a = F/m
    set1 = []
    for _ in range(n_per_law):
        F = rng.uniform(1.0, 20.0)
        m = rng.uniform(0.5, 10.0)
        a = F / m
        set1.append(({"force": F, "mass": m}, a))

    # Set 2: Galilean transform x' = x - v*t (v ≪ c)
    set2 = []
    for _ in range(n_per_law):
        x = rng.uniform(0.0, 100.0)
        v = rng.uniform(0.001, 0.05)   # v ≪ c = 1.0
        t = rng.uniform(0.1, 50.0)
        xprime = x - v * t
        set2.append(({"x": x, "v_frame": v, "t": t}, xprime))

    # Set 3: momentum p = m*v
    set3 = []
    for _ in range(n_per_law):
        m = rng.uniform(0.5, 10.0)
        v = rng.uniform(0.001, 0.05)
        p = m * v
        set3.append(({"mass": m, "velocity": v}, p))

    return PhysicsStream(
        name="newtonian_mechanics",
        description="Newtonian kinematics/dynamics at v ≪ c (natural units)",
        observation_sets=[set1, set2, set3],
        c_value=C_NATURAL,
        expected_law="a=F/m, x'=x-vt, p=mv",
    )


# ---------------------------------------------------------------------------
# Stream 2: Electromagnetic wave propagation
# ---------------------------------------------------------------------------

def em_wave_stream(n_obs: int = 15, seed: int = 0) -> PhysicsStream:
    """Stream 2: EM wave speed observations (c as a universal constant).

    One observation set:
      Set 1 — wave speed = wavelength / period = c:
        input:  {"wavelength": λ, "period": T}
        output: λ / T  (always equals c = 1.0 in natural units)

    The system should discover DIV(var("wavelength"), var("period")) with
    residual ≈ 0, and note that all outputs have the same value c = 1.0.
    """
    rng = random.Random(seed)

    set1 = []
    for _ in range(n_obs):
        lam = rng.uniform(0.1, 10.0)
        T = lam / C_NATURAL   # period = wavelength / c
        # add tiny noise to avoid degenerate OLS
        noise = rng.gauss(0.0, C_NATURAL * 1e-4)
        wave_speed = lam / T + noise   # ≈ c = 1.0
        set1.append(({"wavelength": lam, "period": T}, wave_speed))

    return PhysicsStream(
        name="em_wave",
        description="EM wave propagation: wave_speed = λ/T = c",
        observation_sets=[set1],
        c_value=C_NATURAL,
        expected_law="wave_speed = wavelength / period = c",
    )


# ---------------------------------------------------------------------------
# Stream 3: Michelson-Morley null result
# ---------------------------------------------------------------------------

def michelson_morley_stream(n_obs: int = 12, seed: int = 0) -> PhysicsStream:
    """Stream 3: Michelson-Morley interferometer null results.

    Observation set — fringe shift observed = 0.0 for all orientations:
      input:  {"arm_length": L, "v_earth": v_e, "wavelength": λ}
      output: 0.0  (null result — no fringe shift observed)

    Newtonian + ether prediction (for reference):
      fringe_shift_expected = 2 * L * v_e^2 / (c^2 * λ)

    These are stored in newtonian_predictions for consistency_check.
    """
    rng = random.Random(seed)
    v_earth = 3e-4   # Earth orbital velocity ≈ 30 km/s, in natural units (c=1)

    observations = []
    newton_preds = []

    for _ in range(n_obs):
        L = rng.uniform(1.0, 20.0)         # arm length in metres
        lam = rng.uniform(4e-7, 7e-7)      # visible light wavelength in natural units
        noise = rng.gauss(0.0, 1e-6)       # detector noise
        fringe_obs = 0.0 + noise            # MM null result

        fringe_newton = 2.0 * L * v_earth**2 / (C_NATURAL**2 * lam)

        observations.append(({"arm_length": L, "v_earth": v_earth, "wavelength": lam}, fringe_obs))
        newton_preds.append(({"arm_length": L, "v_earth": v_earth, "wavelength": lam}, fringe_newton))

    return PhysicsStream(
        name="michelson_morley",
        description="MM null result: fringe_shift = 0 (contradicts Newton+ether)",
        observation_sets=[observations],
        newtonian_predictions=[newton_preds],
        c_value=C_NATURAL,
        expected_law="fringe_shift = 0 (all orientations)",
    )


# ---------------------------------------------------------------------------
# Stream 4: Mercury perihelion precession
# ---------------------------------------------------------------------------

def mercury_precession_stream(n_obs: int = 10, seed: int = 0) -> PhysicsStream:
    """Stream 4: Mercury perihelion precession GR anomaly.

    Observation set — measured perihelion advance per orbit:
      input:  {"semi_major_axis": a, "eccentricity": e, "orbital_period": T}
      output: delta_phi  (observed GR residual in arcseconds/century)

    Newtonian prediction: 0.0 residual (closed ellipses after accounting for
    other planets).

    The GR prediction for Mercury's residual:
      delta_phi = 24 * pi^3 * a^2 / (T^2 * c^2 * (1 - e^2))
    In natural units and scaled to arcsec/century:
      ~43 arcsec/century for Mercury's orbit.
    """
    rng = random.Random(seed)

    # Mercury orbital parameters
    a_mercury = 5.791e10   # semi-major axis in metres
    e_mercury = 0.2056     # eccentricity
    T_mercury = 7.6e6      # orbital period in seconds

    # GR perihelion precession formula (radians per orbit):
    # delta_phi_per_orbit = 6*pi*G*M_sun / (a * c^2 * (1-e^2))
    # ≈ 43 arcsec/century for Mercury in SI units
    # We encode the anomaly directly as the float 43.0 arcsec/century

    observations = []
    newton_preds = []

    for _ in range(n_obs):
        # Vary orbital parameters slightly around Mercury's true values
        a = a_mercury * rng.uniform(0.98, 1.02)
        e = e_mercury * rng.uniform(0.95, 1.05)
        T = T_mercury * rng.uniform(0.98, 1.02)

        # Observed: GR residual ≈ 43 arcsec/century + small noise
        delta_phi_obs = _MERCURY_GR_ARCSEC_PER_CENTURY + rng.gauss(0.0, 0.5)

        # Newtonian prediction: 0 (no residual from Newton's law alone)
        delta_phi_newton = 0.0

        observations.append(({"semi_major_axis": a, "eccentricity": e, "orbital_period": T}, delta_phi_obs))
        newton_preds.append(({"semi_major_axis": a, "eccentricity": e, "orbital_period": T}, delta_phi_newton))

    return PhysicsStream(
        name="mercury_precession",
        description="Mercury perihelion: ~43 arcsec/century GR anomaly",
        observation_sets=[observations],
        newtonian_predictions=[newton_preds],
        c_value=C_NATURAL,
        expected_law="delta_phi = 43 arcsec/century (GR correction)",
    )


# ---------------------------------------------------------------------------
# Lorentz factor stream (used in Phase 9 γ(v) recovery test)
# ---------------------------------------------------------------------------

def lorentz_factor_stream(
    c: float = C_NATURAL,
    n_obs: int = 15,
    seed: int = 0,
) -> PhysicsStream:
    """Time-dilation observations for Lorentz factor recovery.

    Observation set — time dilation factor γ(v) = 1/√(1 - v²/c²):
      input:  {"velocity": v}
      output: 1 / sqrt(1 - (v/c)^2)

    This stream is used by the Phase 9 unit test (blocker 1) to verify that
    discover_law can recover the Lorentz factor expression at depth ≥ 4.

    With c=1.0 (natural units), γ(v) = 1/√(1-v²), which has no free parameters
    and should be recovered as:
      DIV(1.0, SQRT(SUB(1.0, SQ(v))))  — tree depth 4, 0 free params.

    Parameters
    ----------
    c     : speed of light (default 1.0 = natural units).
    n_obs : number of velocity samples.
    seed  : random seed for velocity sampling.
    """
    rng = random.Random(seed)
    c_sq_inv = 1.0 / (c * c)

    observations = []
    for _ in range(n_obs):
        v = rng.uniform(0.05, 0.95) * c   # 5%–95% of c
        gamma = 1.0 / math.sqrt(1.0 - (v / c) ** 2)
        observations.append(({"velocity": v}, gamma))

    return PhysicsStream(
        name="lorentz_factor",
        description=f"Lorentz factor γ(v) = 1/sqrt(1-v²/c²), c={c}",
        observation_sets=[observations],
        c_value=c,
        expected_law="gamma = 1/sqrt(1 - v^2/c^2)",
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def all_physics_streams(seed: int = 0) -> list[PhysicsStream]:
    """Return all four Einstein test streams plus the Lorentz factor stream."""
    return [
        newtonian_mechanics_stream(seed=seed),
        em_wave_stream(seed=seed),
        michelson_morley_stream(seed=seed),
        mercury_precession_stream(seed=seed),
        lorentz_factor_stream(seed=seed),
    ]
