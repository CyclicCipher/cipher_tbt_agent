"""Physics world simulator.

Hidden rules (the learner must discover these):
- F = ma (force = mass * acceleration)
- Constant velocity: x(t) = x0 + v*t
- Gravity: F_grav = m * g (g = 9.81 m/s^2, downward)
- Momentum conservation: m1*v1 + m2*v2 = const in collisions
- Energy conservation: KE + PE = const (no friction)

The learner sees: positions, velocities, masses over time.
It must discover: Newton's laws, conservation laws.

Tier 1: predict position from constant velocity
Tier 2: predict trajectory under gravity
Tier 3: discover momentum conservation from collision data
Tier 4: unify falling + orbits (same force law)
Tier 5: discover general relativity (beyond this simulator)
"""
from __future__ import annotations

import math

from .base import World, Observation


class PhysicsWorld(World):
    """A 2D Newtonian physics simulator."""

    GRAVITY = 9.81  # m/s^2
    DT = 0.1        # time step in seconds

    def __init__(self):
        super().__init__()
        # Particles: {name: {mass, x, y, vx, vy}}.
        self._particles: dict[str, dict[str, float]] = {}
        self._time: float = 0.0
        self._forces: dict[str, tuple[float, float]] = {}  # name -> (fx, fy)
        self._gravity_on: bool = False
        self._initial_state: dict = {}

    def add_particle(self, name: str, mass: float,
                     x: float, y: float,
                     vx: float = 0.0, vy: float = 0.0):
        self._particles[name] = {
            "mass": mass, "x": x, "y": y, "vx": vx, "vy": vy,
        }

    def set_gravity(self, on: bool):
        self._gravity_on = on

    def save_initial_state(self):
        import copy
        self._initial_state = {
            "particles": copy.deepcopy(self._particles),
            "time": self._time,
            "gravity_on": self._gravity_on,
        }

    def reset(self):
        import copy
        if self._initial_state:
            self._particles = copy.deepcopy(self._initial_state["particles"])
            self._time = self._initial_state["time"]
            self._gravity_on = self._initial_state["gravity_on"]
        else:
            self._particles.clear()
            self._time = 0.0
        self._forces.clear()
        self._history.clear()

    def observe(self) -> Observation:
        facts = set()
        facts.add(("time", (), round(self._time, 3)))

        for name, p in self._particles.items():
            facts.add(("particle", (name,), True))
            facts.add(("mass", (name,), round(p["mass"], 3)))
            facts.add(("pos_x", (name,), round(p["x"], 3)))
            facts.add(("pos_y", (name,), round(p["y"], 3)))
            facts.add(("vel_x", (name,), round(p["vx"], 3)))
            facts.add(("vel_y", (name,), round(p["vy"], 3)))

            # Derived quantities.
            speed = math.sqrt(p["vx"]**2 + p["vy"]**2)
            ke = 0.5 * p["mass"] * speed**2
            pe = p["mass"] * self.GRAVITY * p["y"] if self._gravity_on else 0.0
            momentum_x = p["mass"] * p["vx"]
            momentum_y = p["mass"] * p["vy"]

            facts.add(("speed", (name,), round(speed, 3)))
            facts.add(("kinetic_energy", (name,), round(ke, 3)))
            facts.add(("potential_energy", (name,), round(pe, 3)))
            facts.add(("total_energy", (name,), round(ke + pe, 3)))
            facts.add(("momentum_x", (name,), round(momentum_x, 3)))
            facts.add(("momentum_y", (name,), round(momentum_y, 3)))

        # System totals (for conservation law discovery).
        total_momentum_x = sum(p["mass"] * p["vx"] for p in self._particles.values())
        total_momentum_y = sum(p["mass"] * p["vy"] for p in self._particles.values())
        total_energy = sum(
            0.5 * p["mass"] * (p["vx"]**2 + p["vy"]**2) +
            (p["mass"] * self.GRAVITY * p["y"] if self._gravity_on else 0.0)
            for p in self._particles.values()
        )
        facts.add(("total_momentum_x", (), round(total_momentum_x, 3)))
        facts.add(("total_momentum_y", (), round(total_momentum_y, 3)))
        facts.add(("total_energy", (), round(total_energy, 3)))

        return Observation(facts=frozenset(facts))

    def available_actions(self) -> list[tuple[str, tuple[str, ...]]]:
        actions = []
        # Time step (advance simulation).
        actions.append(("step", ()))
        # Apply force to a particle.
        for name in self._particles:
            for fx in [-10.0, -5.0, 0.0, 5.0, 10.0]:
                for fy in [-10.0, -5.0, 0.0, 5.0, 10.0]:
                    if fx != 0.0 or fy != 0.0:
                        actions.append(("apply_force", (name, str(fx), str(fy))))
        return actions

    def _execute_impl(self, action: str, args: tuple[str, ...]):
        if action == "step":
            self._step()
        elif action == "apply_force":
            name, fx_str, fy_str = args
            fx, fy = float(fx_str), float(fy_str)
            self._forces[name] = (
                self._forces.get(name, (0.0, 0.0))[0] + fx,
                self._forces.get(name, (0.0, 0.0))[1] + fy,
            )

    def _step(self):
        """Advance one time step using Euler integration."""
        dt = self.DT

        # Check for collisions (simple 1D elastic collision).
        names = list(self._particles.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = self._particles[names[i]], self._particles[names[j]]
                dx = a["x"] - b["x"]
                dy = a["y"] - b["y"]
                dist = math.sqrt(dx**2 + dy**2)
                if dist < 0.5:  # collision threshold
                    self._elastic_collision(names[i], names[j])

        # Update velocities from forces.
        for name, p in self._particles.items():
            fx, fy = self._forces.get(name, (0.0, 0.0))
            if self._gravity_on:
                fy -= p["mass"] * self.GRAVITY
            ax = fx / p["mass"]
            ay = fy / p["mass"]
            p["vx"] += ax * dt
            p["vy"] += ay * dt

        # Update positions.
        for p in self._particles.values():
            p["x"] += p["vx"] * dt
            p["y"] += p["vy"] * dt

            # Floor at y=0.
            if p["y"] < 0:
                p["y"] = 0
                p["vy"] = -p["vy"] * 0.8  # bounce with energy loss

        # Clear forces (they're impulses, not persistent).
        self._forces.clear()
        self._time += dt

    def _elastic_collision(self, name_a: str, name_b: str):
        """1D elastic collision (momentum + energy conserving)."""
        a = self._particles[name_a]
        b = self._particles[name_b]
        m1, m2 = a["mass"], b["mass"]

        # 1D elastic collision formula along x.
        v1x = ((m1 - m2) * a["vx"] + 2 * m2 * b["vx"]) / (m1 + m2)
        v2x = ((m2 - m1) * b["vx"] + 2 * m1 * a["vx"]) / (m1 + m2)
        a["vx"] = v1x
        b["vx"] = v2x

        # Same for y.
        v1y = ((m1 - m2) * a["vy"] + 2 * m2 * b["vy"]) / (m1 + m2)
        v2y = ((m2 - m1) * b["vy"] + 2 * m1 * a["vy"]) / (m1 + m2)
        a["vy"] = v1y
        b["vy"] = v2y


# ---------------------------------------------------------------------------
# Preset scenarios
# ---------------------------------------------------------------------------

def constant_velocity_scenario() -> PhysicsWorld:
    """Tier 1: particle moving at constant velocity (no forces)."""
    w = PhysicsWorld()
    w.add_particle("ball", mass=1.0, x=0.0, y=5.0, vx=3.0, vy=0.0)
    w.save_initial_state()
    return w


def projectile_scenario() -> PhysicsWorld:
    """Tier 2: projectile under gravity."""
    w = PhysicsWorld()
    w.add_particle("ball", mass=1.0, x=0.0, y=0.0, vx=10.0, vy=20.0)
    w.set_gravity(True)
    w.save_initial_state()
    return w


def collision_scenario() -> PhysicsWorld:
    """Tier 3: two particles colliding (momentum conservation)."""
    w = PhysicsWorld()
    w.add_particle("a", mass=2.0, x=0.0, y=5.0, vx=5.0, vy=0.0)
    w.add_particle("b", mass=1.0, x=3.0, y=5.0, vx=-2.0, vy=0.0)
    w.save_initial_state()
    return w


if __name__ == "__main__":
    print("=== Physics World: Constant Velocity ===")
    w = constant_velocity_scenario()
    for i in range(5):
        obs = w.observe()
        x = obs.get("pos_x", ("ball",))
        y = obs.get("pos_y", ("ball",))
        t = obs.get("time", ())
        print(f"  t={t:.1f} x={x:.1f} y={y:.1f}")
        w.execute("step", ())

    print("\n=== Physics World: Collision ===")
    w = collision_scenario()
    for i in range(10):
        obs = w.observe()
        mom_x = obs.get("total_momentum_x", ())
        mom_y = obs.get("total_momentum_y", ())
        print(f"  t={obs.get('time', ()):.1f} total_p=({mom_x:.2f}, {mom_y:.2f})")
        w.execute("step", ())
